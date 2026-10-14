// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Marking waits on I/O outside the process.
//!
//! Code that awaits a remote system (an HTTP request, a call to a hosted
//! model) marks the wait with [`IoWaitExt::io_wait`]. DataFusion only reports
//! when such a wait starts and ends; what a wait means is decided by the
//! [`IoWaitHook`] the application sets with [`set_io_wait_hook`]. A
//! distributed executor can, for example, lend a task's slot to another task
//! while every part of that task is waiting on I/O.
//!
//! Without a hook, [`IoWaitExt::io_wait`] does nothing.

use futures::future::{BoxFuture, Either, LocalBoxFuture};
use futures::{FutureExt, Stream};
use pin_project_lite::pin_project;
use std::any::Any;
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll};

/// Receives the start and end of every [`IoWait`], and follows the work a
/// task spawns through [`SpawnedTask`](crate::SpawnedTask) and
/// [`JoinSet`](crate::JoinSet).
///
/// Every [`before_wait`](Self::before_wait) is followed by exactly one
/// [`after_wait`](Self::after_wait) from the same [`IoWait`], on the same
/// task, including when the wait is dropped while pending.
pub trait IoWaitHook: Send + Sync + 'static {
    /// A wait started: an [`IoWait`] returned `Pending` that was not a
    /// cooperative yield.
    fn before_wait(&self);

    /// That wait ended: the inner future or stream returned `Ready`, or the
    /// [`IoWait`] was dropped while pending.
    fn after_wait(&self);

    /// Wraps a future spawned by [`SpawnedTask`](crate::SpawnedTask) or
    /// [`JoinSet`](crate::JoinSet), so the hook can follow the spawning task
    /// into it. Called on the spawning task, before the future is spawned.
    /// The hook must not change the future's output.
    fn carry_future(
        &self,
        fut: BoxFuture<'static, Box<dyn Any + Send>>,
    ) -> BoxFuture<'static, Box<dyn Any + Send>>;

    /// Wraps a future spawned on a [`LocalSet`](tokio::task::LocalSet) by
    /// [`JoinSet::spawn_local`](crate::JoinSet::spawn_local) or
    /// [`JoinSet::spawn_local_on`](crate::JoinSet::spawn_local_on), as
    /// [`carry_future`](Self::carry_future) does for `Send` futures.
    fn carry_local_future(
        &self,
        fut: LocalBoxFuture<'static, Box<dyn Any>>,
    ) -> LocalBoxFuture<'static, Box<dyn Any>>;

    /// Wraps a blocking closure spawned by [`SpawnedTask`](crate::SpawnedTask)
    /// or [`JoinSet`](crate::JoinSet), as [`carry_future`](Self::carry_future)
    /// does for futures. The hook must not change the closure's return value.
    fn carry_block(
        &self,
        f: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
    ) -> Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>;
}

/// Error returned by [`set_io_wait_hook`].
#[derive(Debug)]
pub enum IoWaitHookError {
    /// The global hook has already been set.
    AlreadySet,
}

impl Display for IoWaitHookError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            IoWaitHookError::AlreadySet => {
                write!(f, "The global IoWaitHook is already set")
            }
        }
    }
}

impl Error for IoWaitHookError {}

static GLOBAL_HOOK: OnceLock<&'static dyn IoWaitHook> = OnceLock::new();

#[inline]
fn get_hook() -> Option<&'static dyn IoWaitHook> {
    GLOBAL_HOOK.get().copied()
}

/// Set the hook that receives every [`IoWait`] and every spawn.
///
/// This should be called once at startup by the application, which decides
/// what a wait means. If called more than once, an
/// `Err(IoWaitHookError::AlreadySet)` is returned. If never called,
/// [`IoWaitExt::io_wait`] does nothing.
pub fn set_io_wait_hook(hook: &'static dyn IoWaitHook) -> Result<(), IoWaitHookError> {
    GLOBAL_HOOK
        .set(hook)
        .map_err(|_set_err| IoWaitHookError::AlreadySet)
}

/// Passes a future about to be spawned through [`IoWaitHook::carry_future`].
/// Without a hook, the future is returned as it is.
pub(crate) fn carry_future<T, F>(future: F) -> impl Future<Output = T> + Send + 'static
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let Some(hook) = get_hook() else {
        return Either::Left(future);
    };
    // Called now, on the spawning task, so the hook can capture its context.
    let carried = hook.carry_future(Box::pin(async move {
        Box::new(future.await) as Box<dyn Any + Send>
    }));
    Either::Right(carried.map(|out| {
        *out.downcast::<T>()
            .expect("IoWaitHook must preserve the future's output type!")
    }))
}

/// Passes a non-`Send` future about to be spawned through
/// [`IoWaitHook::carry_local_future`]. Without a hook, the future is returned
/// as it is.
pub(crate) fn carry_local_future<T, F>(future: F) -> impl Future<Output = T> + 'static
where
    F: Future<Output = T> + 'static,
    T: 'static,
{
    let Some(hook) = get_hook() else {
        return Either::Left(future);
    };
    let carried = hook.carry_local_future(Box::pin(async move {
        Box::new(future.await) as Box<dyn Any>
    }));
    Either::Right(carried.map(|out| {
        *out.downcast::<T>()
            .expect("IoWaitHook must preserve the future's output type!")
    }))
}

/// Passes a blocking closure about to be spawned through
/// [`IoWaitHook::carry_block`]. Without a hook, the closure is returned as it
/// is.
pub(crate) fn carry_block<T, F>(f: F) -> Box<dyn FnOnce() -> T + Send>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let Some(hook) = get_hook() else {
        return Box::new(f);
    };
    let carried =
        hook.carry_block(Box::new(move || Box::new(f()) as Box<dyn Any + Send>));
    Box::new(move || {
        *carried()
            .downcast::<T>()
            .expect("IoWaitHook must preserve the closure's return type!")
    })
}

/// Extension that marks a future or stream as waiting on I/O outside the
/// process, such as a request to a remote service.
///
/// Mark only waits that use no local CPU: a `Pending` that means work running
/// elsewhere in the process (a spawned task, a local model) must stay
/// unmarked.
///
/// ```
/// use datafusion_common_runtime::IoWaitExt;
///
/// # async fn example() {
/// async fn send_request() -> u32 { 200 }
/// let status = send_request().io_wait().await;
/// # assert_eq!(status, 200);
/// # }
/// ```
pub trait IoWaitExt: Sized {
    /// Marks `self` as waiting on I/O outside the process.
    fn io_wait(self) -> IoWait<Self> {
        IoWait {
            inner: self,
            waiting: false,
        }
    }
}

impl<T> IoWaitExt for T {}

pin_project! {
    /// A future or stream marked with [`IoWaitExt::io_wait`].
    ///
    /// For a future, the wait lasts from its first `Pending` until it is
    /// `Ready`. For a stream, each item it is pending on is one wait.
    #[must_use = "futures and streams do nothing unless polled"]
    pub struct IoWait<T> {
        #[pin]
        inner: T,
        waiting: bool,
    }

    impl<T> PinnedDrop for IoWait<T> {
        fn drop(this: Pin<&mut Self>) {
            if this.waiting
                && let Some(hook) = get_hook()
            {
                hook.after_wait();
            }
        }
    }
}

/// Reports the start or end of a wait for one poll of an [`IoWait`].
#[inline]
fn track<T>(waiting: &mut bool, poll: &Poll<T>) {
    let Some(hook) = get_hook() else {
        return;
    };
    match (poll, *waiting) {
        // A `Pending` from an exhausted coop budget is a yield, not a wait:
        // the task is woken again straight away.
        (Poll::Pending, false) if tokio::task::coop::has_budget_remaining() => {
            hook.before_wait();
            *waiting = true;
        }
        (Poll::Ready(_), true) => {
            hook.after_wait();
            *waiting = false;
        }
        _ => {}
    }
}

impl<T: Future> Future for IoWait<T> {
    type Output = T::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T::Output> {
        let this = self.project();
        let poll = this.inner.poll(cx);
        track(this.waiting, &poll);
        poll
    }
}

impl<T: Stream> Stream for IoWait<T> {
    type Item = T::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T::Item>> {
        let this = self.project();
        let poll = this.inner.poll_next(cx);
        track(this.waiting, &poll);
        poll
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JoinSet, SpawnedTask};
    use futures::StreamExt;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, LazyLock};
    use std::time::Duration;

    /// What the test hook saw for one test, recorded through a task-local so
    /// tests running in parallel don't see each other's events.
    #[derive(Default, Debug)]
    struct Events {
        before: AtomicUsize,
        after: AtomicUsize,
        carried_futures: AtomicUsize,
        carried_blocks: AtomicUsize,
    }

    impl Events {
        fn counts(&self) -> [usize; 4] {
            [
                self.before.load(Ordering::SeqCst),
                self.after.load(Ordering::SeqCst),
                self.carried_futures.load(Ordering::SeqCst),
                self.carried_blocks.load(Ordering::SeqCst),
            ]
        }
    }

    tokio::task_local! { static EVENTS: Arc<Events>; }

    thread_local! {
        static BLOCK_EVENTS: std::cell::RefCell<Option<Arc<Events>>> = const { std::cell::RefCell::new(None) };
    }

    fn current_events() -> Option<Arc<Events>> {
        EVENTS
            .try_with(Arc::clone)
            .ok()
            .or_else(|| BLOCK_EVENTS.with(|e| e.borrow().clone()))
    }

    struct TestHook;

    impl IoWaitHook for TestHook {
        fn before_wait(&self) {
            if let Some(e) = current_events() {
                e.before.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn after_wait(&self) {
            if let Some(e) = current_events() {
                e.after.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn carry_future(
            &self,
            fut: BoxFuture<'static, Box<dyn Any + Send>>,
        ) -> BoxFuture<'static, Box<dyn Any + Send>> {
            match EVENTS.try_with(Arc::clone) {
                Ok(e) => {
                    e.carried_futures.fetch_add(1, Ordering::SeqCst);
                    EVENTS.scope(e, fut).boxed()
                }
                Err(_) => fut,
            }
        }

        fn carry_local_future(
            &self,
            fut: LocalBoxFuture<'static, Box<dyn Any>>,
        ) -> LocalBoxFuture<'static, Box<dyn Any>> {
            match EVENTS.try_with(Arc::clone) {
                Ok(e) => {
                    e.carried_futures.fetch_add(1, Ordering::SeqCst);
                    EVENTS.scope(e, fut).boxed_local()
                }
                Err(_) => fut,
            }
        }

        fn carry_block(
            &self,
            f: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
        ) -> Box<dyn FnOnce() -> Box<dyn Any + Send> + Send> {
            match EVENTS.try_with(Arc::clone) {
                Ok(e) => {
                    e.carried_blocks.fetch_add(1, Ordering::SeqCst);
                    Box::new(move || {
                        BLOCK_EVENTS.with(|b| *b.borrow_mut() = Some(e));
                        let out = f();
                        BLOCK_EVENTS.with(|b| *b.borrow_mut() = None);
                        out
                    })
                }
                Err(_) => f,
            }
        }
    }

    static TEST_HOOK: TestHook = TestHook;
    static INSTALL: LazyLock<()> = LazyLock::new(|| {
        set_io_wait_hook(&TEST_HOOK).expect("only the tests set the hook");
    });

    /// Runs `f` with a fresh event recorder in scope and returns what the
    /// hook saw as `[before, after, carried_futures, carried_blocks]`.
    async fn record<F: Future>(f: F) -> [usize; 4] {
        LazyLock::force(&INSTALL);
        let events = Arc::new(Events::default());
        EVENTS.scope(Arc::clone(&events), f).await;
        events.counts()
    }

    async fn remote_call() -> u32 {
        tokio::time::sleep(Duration::from_millis(5)).await;
        7
    }

    #[test]
    fn set_twice_is_an_error() {
        LazyLock::force(&INSTALL);
        let err = set_io_wait_hook(&TEST_HOOK).expect_err("hook already set");
        assert_eq!(err.to_string(), "The global IoWaitHook is already set");
    }

    #[tokio::test]
    async fn pending_future_is_one_wait() {
        let counts = record(async {
            assert_eq!(remote_call().io_wait().await, 7);
        })
        .await;
        assert_eq!(counts, [1, 1, 0, 0]);
    }

    #[tokio::test]
    async fn ready_future_is_no_wait() {
        let counts = record(async {
            assert_eq!(async { 7 }.io_wait().await, 7);
        })
        .await;
        assert_eq!(counts, [0, 0, 0, 0]);
    }

    #[tokio::test]
    async fn spurious_polls_do_not_restart_a_wait() {
        // `join!` polls both futures on every wake, so the slow call is polled
        // while still pending each time the fast ones complete.
        let counts = record(async {
            let slow = async {
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
            .io_wait();
            let fast = async {
                for _ in 0..3 {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            };
            futures::join!(slow, fast);
        })
        .await;
        assert_eq!(counts, [1, 1, 0, 0]);
    }

    #[tokio::test]
    async fn dropped_while_pending_ends_the_wait() {
        let counts = record(async {
            let timed_out =
                tokio::time::timeout(Duration::from_millis(1), remote_call().io_wait())
                    .await;
            assert!(timed_out.is_err(), "remote call should time out");
        })
        .await;
        assert_eq!(counts, [1, 1, 0, 0]);
    }

    #[tokio::test]
    async fn stream_is_one_wait_per_pending_item() {
        let counts = record(async {
            let pages = futures::stream::iter(0..3).then(|_| remote_call());
            assert_eq!(pages.io_wait().collect::<Vec<_>>().await, vec![7, 7, 7]);
        })
        .await;
        assert_eq!(counts, [3, 3, 0, 0]);
    }

    #[tokio::test]
    async fn future_draining_a_stream_is_one_wait() {
        let counts = record(async {
            let pages = futures::stream::iter(0..3).then(|_| remote_call());
            assert_eq!(pages.collect::<Vec<_>>().io_wait().await, vec![7, 7, 7]);
        })
        .await;
        assert_eq!(counts, [1, 1, 0, 0]);
    }

    #[tokio::test]
    async fn cooperative_yields_are_not_waits() {
        let counts = record(async {
            // Each item spends coop budget, so the stream yields `Pending`
            // every time the budget runs out, without any I/O.
            let n = futures::stream::iter(0..2_000)
                .then(|i| async move {
                    tokio::task::consume_budget().await;
                    i
                })
                .io_wait()
                .count()
                .await;
            assert_eq!(n, 2_000);
        })
        .await;
        assert_eq!(counts, [0, 0, 0, 0]);
    }

    #[tokio::test]
    async fn spawned_task_carries_the_hook() {
        let counts = record(async {
            let task = SpawnedTask::spawn(remote_call().io_wait());
            assert_eq!(task.join().await.expect("spawned task"), 7);
        })
        .await;
        assert_eq!(counts, [1, 1, 1, 0]);
    }

    #[tokio::test]
    async fn join_set_carries_the_hook() {
        let counts = record(async {
            let mut set = JoinSet::new();
            for _ in 0..2 {
                set.spawn(remote_call().io_wait());
            }
            let mut total = 0;
            while let Some(r) = set.join_next().await {
                total += r.expect("joined task");
            }
            assert_eq!(total, 14);
        })
        .await;
        assert_eq!(counts, [2, 2, 2, 0]);
    }

    #[tokio::test]
    async fn join_set_local_spawns_carry_the_hook() {
        LazyLock::force(&INSTALL);
        let events = Arc::new(Events::default());
        let local = tokio::task::LocalSet::new();
        let mut set = JoinSet::new();
        // Spawned inside the recorder's scope and polled outside it, so only
        // a carried task reports to the recorder.
        EVENTS.sync_scope(Arc::clone(&events), || {
            set.spawn_local_on(remote_call().io_wait(), &local);
        });
        local
            .run_until(async {
                EVENTS.sync_scope(Arc::clone(&events), || {
                    set.spawn_local(remote_call().io_wait());
                });
                let mut total = 0;
                while let Some(r) = set.join_next().await {
                    total += r.expect("joined task");
                }
                assert_eq!(total, 14);
            })
            .await;
        assert_eq!(events.counts(), [2, 2, 2, 0]);
    }

    #[tokio::test]
    async fn spawned_blocking_closure_is_carried() {
        let seen = Arc::new(Mutex::new(None));
        let seen_in_closure = Arc::clone(&seen);
        let counts = record(async move {
            let task = SpawnedTask::spawn_blocking(move || {
                *seen_in_closure.lock().expect("lock") = Some(current_events().is_some());
                3
            });
            assert_eq!(task.join().await.expect("blocking task"), 3);
        })
        .await;
        assert_eq!(counts, [0, 0, 0, 1]);
        assert_eq!(
            *seen.lock().expect("lock"),
            Some(true),
            "the closure should run inside the carried scope"
        );
    }

    #[tokio::test]
    async fn outside_a_scope_the_hook_sees_nothing() {
        LazyLock::force(&INSTALL);
        // No recorder in scope: the hook is set, but has nothing to report to.
        assert_eq!(remote_call().io_wait().await, 7);
        let task = SpawnedTask::spawn(remote_call().io_wait());
        assert_eq!(task.join().await.expect("spawned task"), 7);
    }
}
