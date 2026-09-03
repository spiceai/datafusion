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

use arrow::datatypes::{DataType, Field, Schema};

use datafusion_common::{
    Column, DFSchema, DFSchemaRef, DataFusionError, Result, TableReference,
    assert_contains,
};
use datafusion_expr::expr::{WindowFunction, WindowFunctionParams};
use datafusion_expr::test::function_stub::{
    count, count_udaf, max, max_udaf, min_udaf, sum, sum_udaf,
};
use datafusion_expr::{
    ColumnarValue, EmptyRelation, Expr, Extension, LogicalPlan, LogicalPlanBuilder,
    ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Union,
    UserDefinedLogicalNode, UserDefinedLogicalNodeCore, Volatility, WindowFrame,
    WindowFunctionDefinition, cast, col, exists, in_subquery, lit, out_ref_col,
    scalar_subquery, table_scan, wildcard,
};
use datafusion_functions::unicode;
use datafusion_functions_aggregate::grouping::grouping_udaf;
use datafusion_functions_nested::make_array::make_array_udf;
use datafusion_functions_nested::map::map_udf;
use datafusion_functions_window::rank::rank_udwf;
use datafusion_functions_window::row_number::row_number_udwf;
use datafusion_sql::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion_sql::unparser::dialect::{
    BigQueryDialect, CustomDialectBuilder, DefaultDialect as UnparserDefaultDialect,
    DefaultDialect, Dialect as UnparserDialect, MySqlDialect as UnparserMySqlDialect,
    PostgreSqlDialect as UnparserPostgreSqlDialect, SnowflakeDialect, SqliteDialect,
};
use datafusion_sql::unparser::{Unparser, expr_to_sql, plan_to_sql};
use insta::assert_snapshot;
use sqlparser::ast::{Ident, ObjectName, Statement};
use std::hash::Hash;
use std::ops::Add;
use std::sync::Arc;
use std::{fmt, vec};

use crate::common::{MockContextProvider, MockSessionState};
use datafusion_expr::builder::{
    project, subquery_alias, table_scan_with_filter_and_fetch, table_scan_with_filters,
};
use datafusion_functions::core::planner::CoreFunctionPlanner;
use datafusion_functions::unicode::planner::UnicodeFunctionPlanner;
use datafusion_functions_nested::extract::array_element_udf;
use datafusion_functions_nested::planner::{FieldAccessPlanner, NestedFunctionPlanner};
use datafusion_sql::unparser::ast::{
    DerivedRelationBuilder, QueryBuilder, RelationBuilder, SelectBuilder,
    TableRelationBuilder,
};
use datafusion_sql::unparser::extension_unparser::{
    UnparseToStatementResult, UnparseWithinStatementResult,
    UserDefinedLogicalNodeUnparser,
};
use sqlparser::dialect::{Dialect, GenericDialect, MySqlDialect};
use sqlparser::parser::Parser;

#[test]
fn test_roundtrip_expr_1() {
    let expr = roundtrip_expr(TableReference::bare("person"), "age > 35").unwrap();
    assert_snapshot!(expr, @"(age > 35)");
}

#[test]
fn test_roundtrip_expr_2() {
    let expr = roundtrip_expr(TableReference::bare("person"), "id = '10'").unwrap();
    assert_snapshot!(expr, @"(id = '10')");
}

#[test]
fn test_roundtrip_expr_3() {
    let expr =
        roundtrip_expr(TableReference::bare("person"), "CAST(id AS VARCHAR)").unwrap();
    assert_snapshot!(expr, @"CAST(id AS VARCHAR)");
}

#[test]
fn test_roundtrip_expr_4() {
    let expr = roundtrip_expr(TableReference::bare("person"), "sum((age * 2))").unwrap();
    assert_snapshot!(expr, @"sum((age * 2))");
}

fn roundtrip_expr(table: TableReference, sql: &str) -> Result<String> {
    let dialect = GenericDialect {};
    let sql_expr = Parser::new(&dialect).try_with_sql(sql)?.parse_expr()?;
    let state = MockSessionState::default().with_aggregate_function(sum_udaf());
    let context = MockContextProvider { state };
    let schema = context.get_table_source(table)?.schema();
    let df_schema = DFSchema::try_from(schema)?;
    let sql_to_rel = SqlToRel::new(&context);
    let expr =
        sql_to_rel.sql_to_expr(sql_expr, &df_schema, &mut PlannerContext::new())?;

    let ast = expr_to_sql(&expr)?;

    Ok(ast.to_string())
}

#[test]
fn roundtrip_statement() -> Result<()> {
    let tests: Vec<&str> = vec![
            "select 1;",
            "select 1 limit 0;",
            "select ta.j1_id from j1 ta join (select 1 as j1_id) tb on ta.j1_id = tb.j1_id;",
            "select ta.j1_id from j1 ta join (select 1 as j1_id) tb using (j1_id);",
            "select ta.j1_id from j1 ta join (select 1 as j1_id) tb on ta.j1_id = tb.j1_id where ta.j1_id > 1;",
            "select ta.j1_id from (select 1 as j1_id) ta;",
            "select ta.j1_id from j1 ta;",
            "select ta.j1_id from j1 ta order by ta.j1_id;",
            "select * from j1 ta order by ta.j1_id, ta.j1_string desc;",
            "select * from j1 limit 10;",
            "select ta.j1_id from j1 ta where ta.j1_id > 1;",
            "select ta.j1_id, tb.j2_string from j1 ta join j2 tb on (ta.j1_id = tb.j2_id);",
            "select ta.j1_id, tb.j2_string, tc.j3_string from j1 ta join j2 tb on (ta.j1_id = tb.j2_id) join j3 tc on (ta.j1_id = tc.j3_id);",
            "select * from (select id, first_name from person)",
            "select * from (select id, first_name from (select * from person))",
            "select id, count(*) as cnt from (select id from person) group by id",
            "select (id-1)/2, count(*) / (sum(id/10)-1) as agg_expr from (select (id-1) as id from person) group by id",
            "select CAST(id/2 as VARCHAR) NOT LIKE 'foo*' from person where NOT EXISTS (select ta.j1_id, tb.j2_string from j1 ta join j2 tb on (ta.j1_id = tb.j2_id))",
            r#"select "First Name" from person_quoted_cols"#,
            "select DISTINCT id FROM person",
            "select DISTINCT on (id) id, first_name from person",
            "select DISTINCT on (id) id, first_name from person order by id",
            r#"select id, count("First Name") as cnt from (select id, "First Name" from person_quoted_cols) group by id"#,
            "select id, count(*) as cnt from (select p1.id as id from person p1 inner join person p2 on p1.id=p2.id) group by id",
            "select id, count(*), first_name from person group by first_name, id",
            "select id, sum(age), first_name from person group by first_name, id",
            "select id, count(*), first_name
            from person
            where id!=3 and first_name=='test'
            group by first_name, id
            having count(*)>5 and count(*)<10
            order by count(*)",
            r#"select id, count("First Name") as count_first_name, "Last Name"
            from person_quoted_cols
            where id!=3 and "First Name"=='test'
            group by "Last Name", id
            having count_first_name>5 and count_first_name<10
            order by count_first_name, "Last Name""#,
            r#"select p.id, count("First Name") as count_first_name,
            "Last Name", sum(qp.id/p.id - (select sum(id) from person_quoted_cols) ) / (select count(*) from person)
            from (select id, "First Name", "Last Name" from person_quoted_cols) qp
            inner join (select * from person) p
            on p.id = qp.id
            where p.id!=3 and "First Name"=='test' and qp.id in
            (select id from (select id, count(*) from person group by id having count(*) > 0))
            group by "Last Name", p.id
            having count_first_name>5 and count_first_name<10
            order by count_first_name, "Last Name""#,
            r#"SELECT j1_string as string FROM j1
            UNION ALL
            SELECT j2_string as string FROM j2"#,
            r#"SELECT j1_string as string FROM j1
            UNION ALL
            SELECT j2_string as string FROM j2
            ORDER BY string DESC
            LIMIT 10"#,
            r#"SELECT col1, id FROM (
                SELECT j1_string AS col1, j1_id AS id FROM j1
                UNION ALL
                SELECT j2_string AS col1, j2_id AS id FROM j2
                UNION ALL
                SELECT j3_string AS col1, j3_id AS id FROM j3
            ) AS subquery GROUP BY col1, id ORDER BY col1 ASC, id ASC"#,
            r#"SELECT col1, id FROM (
                SELECT j1_string AS col1, j1_id AS id FROM j1
                UNION
                SELECT j2_string AS col1, j2_id AS id FROM j2
                UNION
                SELECT j3_string AS col1, j3_id AS id FROM j3
            ) AS subquery ORDER BY col1 ASC, id ASC"#,
            "SELECT id, count(*) over (PARTITION BY first_name ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
            last_name, sum(id) over (PARTITION BY first_name ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
            first_name from person",
            r#"SELECT id, count(distinct id) over (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
            sum(id) OVER (PARTITION BY first_name ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) from person"#,
            "SELECT id, sum(id) OVER (PARTITION BY first_name ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING) from person",
            "WITH t1 AS (SELECT j1_id AS id, j1_string name FROM j1), t2 AS (SELECT j2_id AS id, j2_string name FROM j2) SELECT * FROM t1 JOIN t2 USING (id, name)",
            "WITH w1 AS (SELECT 'a' as col), w2 AS (SELECT 'b' as col), w3 as (SELECT 'c' as col) SELECT * FROM w1 UNION ALL SELECT * FROM w2 UNION ALL SELECT * FROM w3",
            "WITH w1 AS (SELECT 'a' as col), w2 AS (SELECT 'b' as col), w3 as (SELECT 'c' as col), w4 as (SELECT 'd' as col) SELECT * FROM w1 UNION ALL SELECT * FROM w2 UNION ALL SELECT * FROM w3 UNION ALL SELECT * FROM w4",
            "WITH w1 AS (SELECT 'a' as col), w2 AS (SELECT 'b' as col) SELECT * FROM w1 JOIN w2 ON w1.col = w2.col UNION ALL SELECT * FROM w1 JOIN w2 ON w1.col = w2.col UNION ALL SELECT * FROM w1 JOIN w2 ON w1.col = w2.col",
            r#"SELECT id, first_name,
            SUM(id) AS total_sum,
            SUM(id) OVER (PARTITION BY first_name ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING) AS moving_sum,
            MAX(SUM(id)) OVER (PARTITION BY first_name ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS max_total
            FROM person JOIN orders ON person.id = orders.customer_id GROUP BY id, first_name"#,
            r#"SELECT id, first_name,
            SUM(id) AS total_sum,
            SUM(id) OVER (PARTITION BY first_name ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING) AS moving_sum,
            MAX(SUM(id)) OVER (PARTITION BY first_name ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS max_total
            FROM (SELECT id, first_name from person) person JOIN (SELECT customer_id FROM orders) orders ON person.id = orders.customer_id GROUP BY id, first_name"#,
            r#"SELECT id, first_name, last_name, customer_id, SUM(id) AS total_sum
            FROM person
            JOIN orders ON person.id = orders.customer_id
            GROUP BY ROLLUP(id, first_name, last_name, customer_id)"#,
            r#"SELECT id, first_name, last_name,
            SUM(id) AS total_sum,
            COUNT(*) AS total_count,
            SUM(id) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
            FROM person
            GROUP BY GROUPING SETS ((id, first_name, last_name), (first_name, last_name), (last_name))"#,
            "SELECT ARRAY[1, 2, 3]",
            "SELECT ARRAY[1, 2, 3][1]",
            "SELECT [1, 2, 3]",
            "SELECT [1, 2, 3][1]",
            "SELECT left[1] FROM array",
            "SELECT {a:1, b:2}",
            "SELECT s.a FROM (SELECT {a:1, b:2} AS s)",
            "SELECT MAP {'a': 1, 'b': 2}"
    ];

    // For each test sql string, we transform as follows:
    // sql -> ast::Statement (s1) -> LogicalPlan (p1) -> ast::Statement (s2) -> LogicalPlan (p2)
    // We test not that s1==s2, but rather p1==p2. This ensures that unparser preserves the logical
    // query information of the original sql string and disreguards other differences in syntax or
    // quoting.
    for query in tests {
        let dialect = GenericDialect {};
        let statement = Parser::new(&dialect)
            .try_with_sql(query)?
            .parse_statement()?;
        let state = MockSessionState::default()
            .with_scalar_function(make_array_udf())
            .with_scalar_function(array_element_udf())
            .with_scalar_function(map_udf())
            .with_aggregate_function(sum_udaf())
            .with_aggregate_function(count_udaf())
            .with_aggregate_function(max_udaf())
            .with_expr_planner(Arc::new(CoreFunctionPlanner::default()))
            .with_expr_planner(Arc::new(NestedFunctionPlanner))
            .with_expr_planner(Arc::new(FieldAccessPlanner));
        let context = MockContextProvider { state };
        let sql_to_rel = SqlToRel::new(&context);
        let plan = sql_to_rel.sql_statement_to_plan(statement).unwrap();

        let roundtrip_statement = plan_to_sql(&plan)?;

        let plan_roundtrip = sql_to_rel
            .sql_statement_to_plan(roundtrip_statement.clone())
            .unwrap();

        assert_eq!(plan, plan_roundtrip);
    }

    Ok(())
}

#[test]
fn roundtrip_crossjoin() -> Result<()> {
    let query = "select j1.j1_id, j2.j2_string from j1, j2";

    let dialect = GenericDialect {};
    let statement = Parser::new(&dialect)
        .try_with_sql(query)?
        .parse_statement()?;

    let state = MockSessionState::default()
        .with_expr_planner(Arc::new(CoreFunctionPlanner::default()));

    let context = MockContextProvider { state };
    let sql_to_rel = SqlToRel::new(&context);
    let plan = sql_to_rel.sql_statement_to_plan(statement).unwrap();

    let roundtrip_statement = plan_to_sql(&plan)?;

    let actual = &roundtrip_statement.to_string();
    println!("roundtrip sql: {actual}");
    println!("plan {}", plan.display_indent());

    let plan_roundtrip = sql_to_rel
        .sql_statement_to_plan(roundtrip_statement)
        .unwrap();
    assert_snapshot!(
        plan_roundtrip,
        @r"
    Projection: j1.j1_id, j2.j2_string
      Cross Join:
        TableScan: j1
        TableScan: j2
    "
    );

    Ok(())
}

#[macro_export]
macro_rules! roundtrip_statement_with_dialect_helper {
    (
        sql: $sql:expr,
        parser_dialect: $parser_dialect:expr,
        unparser_dialect: $unparser_dialect:expr,
        expected: @ $expected:literal $(,)?
    ) => {{
        let statement = Parser::new(&$parser_dialect)
            .try_with_sql($sql)?
            .parse_statement()?;

        let state = MockSessionState::default()
            .with_aggregate_function(max_udaf())
            .with_aggregate_function(min_udaf())
            .with_expr_planner(Arc::new(CoreFunctionPlanner::default()))
            .with_expr_planner(Arc::new(NestedFunctionPlanner))
            .with_expr_planner(Arc::new(FieldAccessPlanner));

        let context = MockContextProvider { state };
        let sql_to_rel = SqlToRel::new(&context);
        let plan = sql_to_rel
            .sql_statement_to_plan(statement)
            .unwrap_or_else(|e| panic!("Failed to parse sql: {}\n{e}", $sql));

        let unparser = Unparser::new(&$unparser_dialect);
        let roundtrip_statement = unparser.plan_to_sql(&plan)?;

        let actual = &roundtrip_statement.to_string();
        insta::assert_snapshot!(actual, @ $expected);
    }};
}

#[test]
fn roundtrip_statement_with_dialect_1() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select min(ta.j1_id) as j1_min from j1 ta order by min(ta.j1_id) limit 10;",
        parser_dialect: MySqlDialect {},
        unparser_dialect: UnparserMySqlDialect {},
        expected: @"SELECT min(`ta`.`j1_id`) AS `j1_min` FROM `j1` AS `ta` ORDER BY `j1_min` ASC LIMIT 10",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_2() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select min(ta.j1_id) as j1_min from j1 ta order by min(ta.j1_id) limit 10;",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT min(ta.j1_id) AS j1_min FROM j1 AS ta ORDER BY j1_min ASC NULLS LAST LIMIT 10",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_3() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select min(ta.j1_id) as j1_min, max(tb.j1_max) from j1 ta, (select distinct max(ta.j1_id) as j1_max from j1 ta order by max(ta.j1_id)) tb order by min(ta.j1_id) limit 10;",
        parser_dialect: MySqlDialect {},
        unparser_dialect: UnparserMySqlDialect {},
        expected: @"SELECT min(`ta`.`j1_id`) AS `j1_min`, max(`tb`.`j1_max`) FROM `j1` AS `ta` CROSS JOIN (SELECT DISTINCT max(`ta`.`j1_id`) AS `j1_max` FROM `j1` AS `ta`) AS `tb` ORDER BY `j1_min` ASC LIMIT 10",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_postgres_any_array_expr() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select left from array where 1 = any(left);",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserPostgreSqlDialect {},
        expected: @r#"SELECT "array"."left" FROM "array" WHERE 1 = ANY("array"."left")"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_4() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select j1_id from (select 1 as j1_id);",
        parser_dialect: MySqlDialect {},
        unparser_dialect: UnparserMySqlDialect {},
        expected: @"SELECT `j1_id` FROM (SELECT 1 AS `j1_id`) AS `derived_projection`",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_5() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select j1_id from (select j1_id from j1 limit 10);",
        parser_dialect: MySqlDialect {},
        unparser_dialect: UnparserMySqlDialect {},
        expected: @"SELECT `j1`.`j1_id` FROM (SELECT `j1`.`j1_id` FROM `j1` LIMIT 10) AS `derived_limit`",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_6() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select ta.j1_id from j1 ta order by j1_id limit 10;",
        parser_dialect: MySqlDialect {},
        unparser_dialect: UnparserMySqlDialect {},
        expected: @"SELECT `ta`.`j1_id` FROM `j1` AS `ta` ORDER BY `ta`.`j1_id` ASC LIMIT 10",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_7() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select ta.j1_id from j1 ta order by j1_id limit 10;",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT ta.j1_id FROM j1 AS ta ORDER BY ta.j1_id ASC NULLS LAST LIMIT 10",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_8() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT j1_id FROM j1
                  UNION ALL
                  SELECT tb.j2_id as j1_id FROM j2 tb
                  ORDER BY j1_id
                  LIMIT 10;",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT j1.j1_id FROM j1 UNION ALL SELECT tb.j2_id AS j1_id FROM j2 AS tb ORDER BY j1_id ASC NULLS LAST LIMIT 10",
    );
    Ok(())
}

// Test query with derived tables that put distinct,sort,limit on the wrong level
#[test]
fn roundtrip_statement_with_dialect_9() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT j1_string from j1 order by j1_id",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT j1.j1_string FROM j1 ORDER BY j1.j1_id ASC NULLS LAST",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_10() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT j1_string AS a from j1 order by j1_id",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT j1.j1_string AS a FROM j1 ORDER BY j1.j1_id ASC NULLS LAST",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_11() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT j1_string from j1 join j2 on j1.j1_id = j2.j2_id order by j1_id",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT j1.j1_string FROM j1 INNER JOIN j2 ON (j1.j1_id = j2.j2_id) ORDER BY j1.j1_id ASC NULLS LAST",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_12() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "
                SELECT
                  j1_string,
                  j2_string
                FROM
                  (
                    SELECT
                      distinct j1_id,
                      j1_string,
                      j2_string
                    from
                      j1
                      INNER join j2 ON j1.j1_id = j2.j2_id
                    order by
                      j1.j1_id desc
                    limit
                      10
                  ) abc
                ORDER BY
                  abc.j2_string",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT abc.j1_string, abc.j2_string FROM (SELECT DISTINCT j1.j1_id, j1.j1_string, j2.j2_string FROM j1 INNER JOIN j2 ON (j1.j1_id = j2.j2_id) ORDER BY j1.j1_id DESC NULLS FIRST LIMIT 10) AS abc ORDER BY abc.j2_string ASC NULLS LAST",
    );
    Ok(())
}

// more tests around subquery/derived table roundtrip
#[test]
fn roundtrip_statement_with_dialect_13() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT string_count FROM (
                    SELECT
                        j1_id,
                        min(j2_string)
                    FROM
                        j1 LEFT OUTER JOIN j2 ON
                                    j1_id = j2_id
                    GROUP BY
                        j1_id
                ) AS agg (id, string_count)
            ",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT agg.string_count FROM (SELECT j1.j1_id, min(j2.j2_string) FROM j1 LEFT OUTER JOIN j2 ON (j1.j1_id = j2.j2_id) GROUP BY j1.j1_id) AS agg (id, string_count)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_14() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "
                SELECT
                  j1_string,
                  j2_string
                FROM
                  (
                    SELECT
                      j1_id,
                      j1_string,
                      j2_string
                    from
                      j1
                      INNER join j2 ON j1.j1_id = j2.j2_id
                    group by
                      j1_id,
                      j1_string,
                      j2_string
                    order by
                      j1.j1_id desc
                    limit
                      10
                  ) abc
                ORDER BY
                  abc.j2_string",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT abc.j1_string, abc.j2_string FROM (SELECT j1.j1_id, j1.j1_string, j2.j2_string FROM j1 INNER JOIN j2 ON (j1.j1_id = j2.j2_id) GROUP BY j1.j1_id, j1.j1_string, j2.j2_string ORDER BY j1.j1_id DESC NULLS FIRST LIMIT 10) AS abc ORDER BY abc.j2_string ASC NULLS LAST",
    );
    Ok(())
}

// Test query that order by columns are not in select columns
#[test]
fn roundtrip_statement_with_dialect_15() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "
                SELECT
                  j1_string
                FROM
                  (
                    SELECT
                      j1_string,
                      j2_string
                    from
                      j1
                      INNER join j2 ON j1.j1_id = j2.j2_id
                    order by
                      j1.j1_id desc,
                      j2.j2_id desc
                    limit
                      10
                  ) abc
                ORDER BY
                  j2_string",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT abc.j1_string FROM (SELECT j1.j1_string, j2.j2_string FROM j1 INNER JOIN j2 ON (j1.j1_id = j2.j2_id) ORDER BY j1.j1_id DESC NULLS FIRST, j2.j2_id DESC NULLS FIRST LIMIT 10) AS abc ORDER BY abc.j2_string ASC NULLS LAST",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_16() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT j1_id from j1) AS c (id)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT j1.j1_id FROM j1) AS c (id)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_17() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT j1_id as id from j1) AS c",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT j1.j1_id AS id FROM j1) AS c",
    );
    Ok(())
}

// Test query that has calculation in derived table with columns
#[test]
fn roundtrip_statement_with_dialect_18() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT j1_id + 1 * 3 from j1) AS c (id)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT (j1.j1_id + (1 * 3)) FROM j1) AS c (id)",
    );
    Ok(())
}

// Test query that has limit/distinct/order in derived table with columns
#[test]
fn roundtrip_statement_with_dialect_19() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT distinct (j1_id + 1 * 3) FROM j1 LIMIT 1) AS c (id)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT DISTINCT (j1.j1_id + (1 * 3)) FROM j1 LIMIT 1) AS c (id)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_20() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT j1_id + 1 FROM j1 ORDER BY j1_id DESC LIMIT 1) AS c (id)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT (j1.j1_id + 1) FROM j1 ORDER BY j1.j1_id DESC NULLS FIRST LIMIT 1) AS c (id)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_21() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT CAST((CAST(j1_id as BIGINT) + 1) as int) * 10 FROM j1 LIMIT 1) AS c (id)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT (CAST((CAST(j1.j1_id AS BIGINT) + 1) AS INTEGER) * 10) FROM j1 LIMIT 1) AS c (id)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_22() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT CAST(j1_id as BIGINT) + 1 FROM j1 ORDER BY j1_id LIMIT 1) AS c (id)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT (CAST(j1.j1_id AS BIGINT) + 1) FROM j1 ORDER BY j1.j1_id ASC NULLS LAST LIMIT 1) AS c (id)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_23() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT temp_j.id2 FROM (SELECT j1_id, j1_string FROM j1) AS temp_j(id2, string2)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT temp_j.id2 FROM (SELECT j1.j1_id, j1.j1_string FROM j1) AS temp_j (id2, string2)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_24() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT temp_j.id2 FROM (SELECT j1_id, j1_string FROM j1) AS temp_j(id2, string2)",
        parser_dialect: GenericDialect {},
        unparser_dialect: SqliteDialect {},
        expected: @"SELECT `temp_j`.`id2` FROM (SELECT `j1`.`j1_id` AS `id2`, `j1`.`j1_string` AS `string2` FROM `j1`) AS `temp_j`",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_25() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM (SELECT j1_id + 1 FROM j1) AS temp_j(id2)",
        parser_dialect: GenericDialect {},
        unparser_dialect: SqliteDialect {},
        expected: @"SELECT `temp_j`.`id2` FROM (SELECT (`j1`.`j1_id` + 1) AS `id2` FROM `j1`) AS `temp_j`",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_26() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM (SELECT j1_id FROM j1 LIMIT 1) AS temp_j(id2)",
        parser_dialect: GenericDialect {},
        unparser_dialect: SqliteDialect {},
        expected: @"SELECT `temp_j`.`id2` FROM (SELECT `j1`.`j1_id` AS `id2` FROM `j1` LIMIT 1) AS `temp_j`",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_27() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3])",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT "UNNEST(make_array(Int64(1),Int64(2),Int64(3)))" FROM (SELECT UNNEST([1, 2, 3]) AS "UNNEST(make_array(Int64(1),Int64(2),Int64(3)))") AS derived_projection ("UNNEST(make_array(Int64(1),Int64(2),Int64(3)))")"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_28() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) AS t1 (c1)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT t1.c1 FROM (SELECT UNNEST([1, 2, 3]) AS "UNNEST(make_array(Int64(1),Int64(2),Int64(3)))") AS t1 (c1)"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_29() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]), j1",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT "UNNEST(make_array(Int64(1),Int64(2),Int64(3)))", j1.j1_id, j1.j1_string FROM (SELECT UNNEST([1, 2, 3]) AS "UNNEST(make_array(Int64(1),Int64(2),Int64(3)))") AS derived_projection ("UNNEST(make_array(Int64(1),Int64(2),Int64(3)))") CROSS JOIN j1"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_30() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) u(c1) JOIN j1 ON u.c1 = j1.j1_id",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT u.c1, j1.j1_id, j1.j1_string FROM (SELECT UNNEST([1, 2, 3]) AS "UNNEST(make_array(Int64(1),Int64(2),Int64(3)))") AS u (c1) INNER JOIN j1 ON (u.c1 = j1.j1_id)"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_31() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) u(c1) UNION ALL SELECT * FROM UNNEST([4,5,6]) u(c1)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT u.c1 FROM (SELECT UNNEST([1, 2, 3]) AS "UNNEST(make_array(Int64(1),Int64(2),Int64(3)))") AS u (c1) UNION ALL SELECT u.c1 FROM (SELECT UNNEST([4, 5, 6]) AS "UNNEST(make_array(Int64(4),Int64(5),Int64(6)))") AS u (c1)"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_32() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3])",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT UNNEST(make_array(Int64(1),Int64(2),Int64(3))) FROM UNNEST([1, 2, 3])",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_33() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM unnest_table u, UNNEST(u.array_col)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT u.array_col, u.struct_col, "UNNEST(outer_ref(u.array_col))" FROM unnest_table AS u CROSS JOIN LATERAL (SELECT UNNEST(u.array_col) AS "UNNEST(outer_ref(u.array_col))")"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_34() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) AS t1 (c1)",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT t1.c1 FROM UNNEST([1, 2, 3]) AS t1 (c1)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_35() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]), j1",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT UNNEST(make_array(Int64(1),Int64(2),Int64(3))), j1.j1_id, j1.j1_string FROM UNNEST([1, 2, 3]) CROSS JOIN j1",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_36() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) u(c1) JOIN j1 ON u.c1 = j1.j1_id",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT u.c1, j1.j1_id, j1.j1_string FROM UNNEST([1, 2, 3]) AS u (c1) INNER JOIN j1 ON (u.c1 = j1.j1_id)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_37() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) u(c1) UNION ALL SELECT * FROM UNNEST([4,5,6]) u(c1)",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT u.c1 FROM UNNEST([1, 2, 3]) AS u (c1) UNION ALL SELECT u.c1 FROM UNNEST([4, 5, 6]) AS u (c1)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_38() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT UNNEST([1,2,3])",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT * FROM UNNEST([1, 2, 3])",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_39() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT UNNEST([1,2,3]) as c1",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT UNNEST([1, 2, 3]) AS c1",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_40() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT UNNEST([1,2,3]), 1",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT UNNEST([1, 2, 3]) AS UNNEST(make_array(Int64(1),Int64(2),Int64(3))), Int64(1)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_41() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM unnest_table u, UNNEST(u.array_col)",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT u.array_col, u.struct_col, UNNEST(outer_ref(u.array_col)) FROM unnest_table AS u CROSS JOIN UNNEST(u.array_col)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_42() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM unnest_table u, UNNEST(u.array_col) AS t1 (c1)",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT u.array_col, u.struct_col, t1.c1 FROM unnest_table AS u CROSS JOIN UNNEST(u.array_col) AS t1 (c1)",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_43() -> Result<(), DataFusionError> {
    let unparser = CustomDialectBuilder::default()
        .with_unnest_as_table_factor(true)
        .build();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT unnest([1, 2, 3, 4]) from unnest([1, 2, 3]);",
        parser_dialect: GenericDialect {},
        unparser_dialect: unparser,
        expected: @"SELECT UNNEST([1, 2, 3, 4]) AS UNNEST(make_array(Int64(1),Int64(2),Int64(3),Int64(4))) FROM UNNEST([1, 2, 3])",
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_45() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM unnest_table u, UNNEST(u.array_col) AS t1 (c1)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT u.array_col, u.struct_col, t1.c1 FROM unnest_table AS u CROSS JOIN LATERAL (SELECT UNNEST(u.array_col) AS "UNNEST(outer_ref(u.array_col))") AS t1 (c1)"#,
    );
    Ok(())
}

#[test]
fn roundtrip_statement_with_dialect_special_char_alias() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select min(a) as \"min(a)\" from (select 1 as a)",
        parser_dialect: GenericDialect {},
        unparser_dialect: BigQueryDialect {},
        expected: @"SELECT min(`a`) AS `min_40a_41` FROM (SELECT 1 AS `a`)",
    );
    roundtrip_statement_with_dialect_helper!(
        sql: "select a as \"a*\", b as \"b@\" from (select 1 as a , 2 as b)",
        parser_dialect: GenericDialect {},
        unparser_dialect: BigQueryDialect {},
        expected: @"SELECT `a` AS `a_42`, `b` AS `b_64` FROM (SELECT 1 AS `a`, 2 AS `b`)",
    );
    roundtrip_statement_with_dialect_helper!(
        sql: "select a as \"a*\", b , c as \"c@\" from (select 1 as a , 2 as b, 3 as c)",
        parser_dialect: GenericDialect {},
        unparser_dialect: BigQueryDialect {},
        expected: @"SELECT `a` AS `a_42`, `b`, `c` AS `c_64` FROM (SELECT 1 AS `a`, 2 AS `b`, 3 AS `c`)",
    );
    roundtrip_statement_with_dialect_helper!(
        sql: "select * from (select a as \"a*\", b as \"b@\" from (select 1 as a , 2 as b)) where \"a*\" = 1",
        parser_dialect: GenericDialect {},
        unparser_dialect: BigQueryDialect {},
        expected: @"SELECT `a_42`, `b_64` FROM (SELECT `a` AS `a_42`, `b` AS `b_64` FROM (SELECT 1 AS `a`, 2 AS `b`)) WHERE (`a_42` = 1)",
    );
    roundtrip_statement_with_dialect_helper!(
        sql: "select * from (select a as \"a*\", b as \"b@\" from (select 1 as a , 2 as b)) where \"a*\" = 1",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @r#"SELECT "a*", "b@" FROM (SELECT a AS "a*", b AS "b@" FROM (SELECT 1 AS a, 2 AS b)) WHERE ("a*" = 1)"#,
    );
    Ok(())
}

#[test]
fn bigquery_unparses_distinct_and_all_unions_explicitly() -> Result<()> {
    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let scan = |name: &'static str| {
        table_scan(Some(name), &schema, None)?
            .project(vec![col(format!("{name}.id"))])?
            .build()
    };

    let left = scan("left_table")?;
    let right = scan("right_table")?;
    let distinct = LogicalPlanBuilder::from(left.clone())
        .union_distinct(right.clone())?
        .build()?;
    let all = LogicalPlanBuilder::from(left).union(right)?.build()?;

    let dialect = BigQueryDialect::new();
    let distinct_sql = Unparser::new(&dialect).plan_to_sql(&distinct)?.to_string();
    let all_sql = Unparser::new(&dialect).plan_to_sql(&all)?.to_string();

    assert!(
        distinct_sql.contains(" UNION DISTINCT "),
        "BigQuery requires an explicit DISTINCT quantifier: {distinct_sql}"
    );
    assert!(
        all_sql.contains(" UNION ALL "),
        "a logical UNION ALL must retain duplicate rows: {all_sql}"
    );
    assert!(
        !all_sql.contains(" UNION DISTINCT "),
        "UNION ALL must not be rewritten as distinct: {all_sql}"
    );

    let default_sql = Unparser::new(&UnparserDefaultDialect {})
        .plan_to_sql(&distinct)?
        .to_string();
    assert!(
        default_sql.contains(" UNION SELECT "),
        "dialects that accept implicit DISTINCT should retain their rendering: {default_sql}"
    );

    Ok(())
}

#[test]
fn bigquery_omits_numbering_frames_and_retains_aggregate_frames() -> Result<()> {
    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let input = table_scan(Some("window_values"), &schema, None)?.build()?;
    let order_by = vec![col("window_values.id").sort(true, true)];
    let row_number = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: order_by.clone(),
            window_frame: WindowFrame::new(Some(false)),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_num");
    let running_sum = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::AggregateUDF(sum_udaf()),
        params: WindowFunctionParams {
            args: vec![col("window_values.id")],
            partition_by: vec![],
            order_by,
            window_frame: WindowFrame::new(Some(true)),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("running_sum");
    let plan = LogicalPlanBuilder::from(input)
        .window(vec![row_number, running_sum])?
        .build()?;

    let dialect = BigQueryDialect::new();
    let sql = Unparser::new(&dialect).plan_to_sql(&plan)?.to_string();
    assert!(
        sql.contains(
            "row_number() OVER (ORDER BY `window_values`.`id` ASC NULLS FIRST) AS `row_num`"
        ),
        "BigQuery rejects a window frame on ROW_NUMBER: {sql}"
    );
    assert!(
        sql.contains(
            "sum(`window_values`.`id`) OVER (ORDER BY `window_values`.`id` ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS `running_sum`"
        ),
        "aggregate window frames change which rows contribute and must be retained: {sql}"
    );

    let start = sqlparser::ast::WindowFrameBound::Preceding(None);
    let end = sqlparser::ast::WindowFrameBound::CurrentRow;
    for function in [
        "row_number",
        "rank",
        "dense_rank",
        "percent_rank",
        "cume_dist",
        "ntile",
    ] {
        assert!(
            !dialect.window_func_support_window_frame(function, &start, &end),
            "BigQuery does not allow a frame on numbering function {function}"
        );
    }
    assert!(
        dialect.window_func_support_window_frame("sum", &start, &end),
        "aggregate functions must retain their frames"
    );

    Ok(())
}

#[test]
fn test_unnest_logical_plan() -> Result<()> {
    let query = "select unnest(struct_col), unnest(array_col), struct_col, array_col from unnest_table";

    let dialect = GenericDialect {};
    let statement = Parser::new(&dialect)
        .try_with_sql(query)?
        .parse_statement()?;

    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let sql_to_rel = SqlToRel::new(&context);
    let plan = sql_to_rel.sql_statement_to_plan(statement).unwrap();
    assert_snapshot!(
        plan,
        @r"
    Projection: __unnest_placeholder(unnest_table.struct_col).field1 AS unnest_table.struct_col.field1, __unnest_placeholder(unnest_table.struct_col).field2 AS unnest_table.struct_col.field2, __unnest_placeholder(unnest_table.array_col,depth=1) AS UNNEST(unnest_table.array_col), unnest_table.struct_col, unnest_table.array_col
      Unnest: lists[__unnest_placeholder(unnest_table.array_col)|depth=1] structs[__unnest_placeholder(unnest_table.struct_col)]
        Projection: unnest_table.struct_col AS __unnest_placeholder(unnest_table.struct_col), unnest_table.array_col AS __unnest_placeholder(unnest_table.array_col), unnest_table.struct_col, unnest_table.array_col
          TableScan: unnest_table
    "
    );

    Ok(())
}

#[test]
fn test_aggregation_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::UInt8, false),
    ]);

    let plan = LogicalPlanBuilder::from(
        table_scan(Some("users"), &schema, Some(vec![0, 1]))?.build()?,
    )
    .aggregate(vec![col("name")], vec![sum(col("age"))])?
    .build()?;

    let unparser = Unparser::default();
    let statement = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        statement,
        @r#"SELECT sum(users.age), users."name" FROM users GROUP BY users."name""#
    );

    Ok(())
}

/// return a schema with two string columns: "id" and "value"
fn test_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
    ])
}

#[test]
fn test_table_references_in_plan_to_sql_1() {
    let table_name = "catalog.schema.table";
    let schema = test_schema();
    let sql = table_references_in_plan_helper(
        table_name,
        schema,
        vec![col("id"), col("value")],
        &DefaultDialect {},
    );
    assert_snapshot!(
        sql,
        @r#"SELECT "table".id, "table"."value" FROM "catalog"."schema"."table""#
    );
}

#[test]
fn test_table_references_in_plan_to_sql_2() {
    let table_name = "schema.table";
    let schema = test_schema();
    let sql = table_references_in_plan_helper(
        table_name,
        schema,
        vec![col("id"), col("value")],
        &DefaultDialect {},
    );
    assert_snapshot!(
        sql,
        @r#"SELECT "table".id, "table"."value" FROM "schema"."table""#
    );
}

#[test]
fn test_table_references_in_plan_to_sql_3() {
    let table_name = "table";
    let schema = test_schema();
    let sql = table_references_in_plan_helper(
        table_name,
        schema,
        vec![col("id"), col("value")],
        &DefaultDialect {},
    );
    assert_snapshot!(
        sql,
        @r#"SELECT "table".id, "table"."value" FROM "table""#
    );
}

#[test]
fn test_table_references_in_plan_to_sql_4() {
    let table_name = "catalog.schema.table";
    let schema = test_schema();
    let custom_dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .with_identifier_quote_style('"')
        .build();

    let sql = table_references_in_plan_helper(
        table_name,
        schema,
        vec![col("id"), col("value")],
        &custom_dialect,
    );
    assert_snapshot!(
        sql,
        @r#"SELECT "catalog"."schema"."table"."id", "catalog"."schema"."table"."value" FROM "catalog"."schema"."table""#
    );
}

#[test]
fn test_table_references_in_plan_to_sql_5() {
    let table_name = "schema.table";
    let schema = test_schema();
    let custom_dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .with_identifier_quote_style('"')
        .build();

    let sql = table_references_in_plan_helper(
        table_name,
        schema,
        vec![col("id"), col("value")],
        &custom_dialect,
    );
    assert_snapshot!(
        sql,
        @r#"SELECT "schema"."table"."id", "schema"."table"."value" FROM "schema"."table""#
    );
}

#[test]
fn test_table_references_in_plan_to_sql_6() {
    let table_name = "table";
    let schema = test_schema();
    let custom_dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .with_identifier_quote_style('"')
        .build();

    let sql = table_references_in_plan_helper(
        table_name,
        schema,
        vec![col("id"), col("value")],
        &custom_dialect,
    );
    assert_snapshot!(
        sql,
        @r#"SELECT "table"."id", "table"."value" FROM "table""#
    );
}

fn table_references_in_plan_helper(
    table_name: &str,
    table_schema: Schema,
    expr: impl IntoIterator<Item = impl Into<datafusion_expr::select_expr::SelectExpr>>,
    dialect: &impl UnparserDialect,
) -> Statement {
    let plan = table_scan(Some(table_name), &table_schema, None)
        .unwrap()
        .project(expr)
        .unwrap()
        .build()
        .unwrap();
    let unparser = Unparser::new(dialect);
    unparser.plan_to_sql(&plan).unwrap()
}

#[test]
fn test_table_scan_with_none_projection_in_plan_to_sql_1() {
    let schema = test_schema();
    let table_name = "catalog.schema.table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name, schema, None,
    );
    let sql = plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM "catalog"."schema"."table""#
    );
}

#[test]
fn test_table_scan_with_none_projection_in_plan_to_sql_2() {
    let schema = test_schema();
    let table_name = "schema.table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name, schema, None,
    );
    let sql = plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM "schema"."table""#
    );
}

#[test]
fn test_table_scan_with_none_projection_in_plan_to_sql_3() {
    let schema = test_schema();
    let table_name = "table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name, schema, None,
    );
    let sql = plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM "table""#
    );
}

#[test]
fn test_table_scan_with_empty_projection_in_plan_to_sql_1() {
    let schema = test_schema();
    let table_name = "catalog.schema.table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name,
        schema,
        Some(vec![]),
    );
    let sql = plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT 1 FROM "catalog"."schema"."table""#
    );
}

#[test]
fn test_table_scan_with_empty_projection_in_plan_to_sql_2() {
    let schema = test_schema();
    let table_name = "schema.table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name,
        schema,
        Some(vec![]),
    );
    let sql = plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT 1 FROM "schema"."table""#
    );
}

#[test]
fn test_table_scan_with_empty_projection_in_plan_to_sql_3() {
    let schema = test_schema();
    let table_name = "table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name,
        schema,
        Some(vec![]),
    );
    let sql = plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT 1 FROM "table""#
    );
}

#[test]
fn test_table_scan_with_empty_projection_in_plan_to_sql_postgres() {
    let schema = test_schema();
    let table_name = "table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name,
        schema,
        Some(vec![]),
    );
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT FROM "table""#
    );
}

#[test]
fn test_table_scan_with_empty_projection_in_plan_to_sql_default_dialect() {
    let schema = test_schema();
    let table_name = "table";
    let plan = table_scan_with_empty_projection_and_none_projection_helper(
        table_name,
        schema,
        Some(vec![]),
    );
    let unparser = Unparser::new(&UnparserDefaultDialect {});
    let sql = unparser.plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT 1 FROM "table""#
    );
}

#[test]
fn test_table_scan_with_empty_projection_and_filter_postgres() {
    let schema = test_schema();
    let table_name = "table";
    let plan = table_scan_with_filter_and_fetch(
        Some(table_name),
        &schema,
        Some(vec![]),
        vec![col("id").gt(lit(10))],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT FROM "table" WHERE ("table"."id" > 10)"#
    );
}

#[test]
fn test_table_scan_with_empty_projection_and_filter_default_dialect() {
    let schema = test_schema();
    let table_name = "table";
    let plan = table_scan_with_filter_and_fetch(
        Some(table_name),
        &schema,
        Some(vec![]),
        vec![col("id").gt(lit(10))],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    let unparser = Unparser::new(&UnparserDefaultDialect {});
    let sql = unparser.plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT 1 FROM "table" WHERE ("table".id > 10)"#
    );
}

fn table_scan_with_empty_projection_and_none_projection_helper(
    table_name: &str,
    table_schema: Schema,
    projection: Option<Vec<usize>>,
) -> LogicalPlan {
    table_scan(Some(table_name), &table_schema, projection)
        .unwrap()
        .build()
        .unwrap()
}

// An empty `Projection` node (0 output expressions) arises when, for example,
// `count(*)` is planned over a view/subquery whose columns are all pruned,
// yielding `Projection: <empty> -> TableScan`. It must not be unparsed as an
// empty `SELECT` list for dialects that reject `SELECT FROM t` (e.g. DuckDB):
// fall back to `SELECT 1` just like the bare empty-projection `TableScan`.
fn empty_projection_over_table_helper(
    table_name: &str,
    table_schema: Schema,
) -> LogicalPlan {
    project(
        table_scan(Some(table_name), &table_schema, None)
            .unwrap()
            .build()
            .unwrap(),
        Vec::<Expr>::new(),
    )
    .unwrap()
}

#[test]
fn test_empty_projection_node_in_plan_to_sql_default_dialect() {
    let plan = empty_projection_over_table_helper("table", test_schema());
    let unparser = Unparser::new(&UnparserDefaultDialect {});
    let sql = unparser.plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT 1 FROM "table""#
    );
}

#[test]
fn test_empty_projection_node_in_plan_to_sql_postgres() {
    let plan = empty_projection_over_table_helper("table", test_schema());
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan).unwrap();
    assert_snapshot!(
        sql,
        @r#"SELECT FROM "table""#
    );
}

#[test]
fn test_empty_projection_under_subquery_alias_default_dialect() {
    let plan = subquery_alias(
        empty_projection_over_table_helper("table", test_schema()),
        "v",
    )
    .unwrap();
    let unparser = Unparser::new(&UnparserDefaultDialect {});
    let sql = unparser.plan_to_sql(&plan).unwrap();
    // The inner empty projection (the view body) is what previously unparsed to
    // the invalid `SELECT FROM "table"`; it now renders as `SELECT 1 FROM "table"`.
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM (SELECT 1 FROM "table") AS v"#
    );
}

#[test]
fn test_pretty_roundtrip() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Utf8, false),
    ]);

    let df_schema = DFSchema::try_from(schema)?;

    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let sql_to_rel = SqlToRel::new(&context);

    let unparser = Unparser::default().with_pretty(true);

    let sql_to_pretty_unparse = vec![
        ("((id < 5) OR (age = 8))", "id < 5 OR age = 8"),
        ("((id + 5) * (age * 8))", "(id + 5) * age * 8"),
        ("(3 + (5 * 6) * 3)", "3 + 5 * 6 * 3"),
        ("((3 * (5 + 6)) * 3)", "3 * (5 + 6) * 3"),
        ("((3 AND (5 OR 6)) * 3)", "(3 AND (5 OR 6)) * 3"),
        ("((3 + (5 + 6)) * 3)", "(3 + 5 + 6) * 3"),
        ("((3 + (5 + 6)) + 3)", "3 + 5 + 6 + 3"),
        ("3 + 5 + 6 + 3", "3 + 5 + 6 + 3"),
        ("3 + (5 + (6 + 3))", "3 + 5 + 6 + 3"),
        ("3 + ((5 + 6) + 3)", "3 + 5 + 6 + 3"),
        ("(3 + 5) + (6 + 3)", "3 + 5 + 6 + 3"),
        ("((3 + 5) + (6 + 3))", "3 + 5 + 6 + 3"),
        (
            "((id > 10) OR (age BETWEEN 10 AND 20))",
            "id > 10 OR age BETWEEN 10 AND 20",
        ),
        (
            "((id > 10) * (age BETWEEN 10 AND 20))",
            "(id > 10) * (age BETWEEN 10 AND 20)",
        ),
        ("id - (age - 8)", "id - (age - 8)"),
        ("((id - age) - 8)", "id - age - 8"),
        ("(id OR (age - 8))", "id OR age - 8"),
        ("(id / (age - 8))", "id / (age - 8)"),
        ("((id / age) * 8)", "id / age * 8"),
        ("((age + 10) < 20) IS TRUE", "(age + 10 < 20) IS TRUE"),
        (
            "(20 > (age + 5)) IS NOT FALSE",
            "(20 > age + 5) IS NOT FALSE",
        ),
        ("(true AND false) IS FALSE", "(true AND false) IS FALSE"),
        ("true AND (false IS FALSE)", "true AND false IS FALSE"),
    ];

    for (sql, pretty) in sql_to_pretty_unparse.iter() {
        let sql_expr = Parser::new(&GenericDialect {})
            .try_with_sql(sql)?
            .parse_expr()?;
        let expr =
            sql_to_rel.sql_to_expr(sql_expr, &df_schema, &mut PlannerContext::new())?;
        let round_trip_sql = unparser.expr_to_sql(&expr)?.to_string();
        assert_eq!((*pretty).to_string(), round_trip_sql);

        // verify that the pretty string parses to the same underlying Expr
        let pretty_sql_expr = Parser::new(&GenericDialect {})
            .try_with_sql(pretty)?
            .parse_expr()?;

        let pretty_expr = sql_to_rel.sql_to_expr(
            pretty_sql_expr,
            &df_schema,
            &mut PlannerContext::new(),
        )?;

        assert_eq!(expr.to_string(), pretty_expr.to_string());
    }

    Ok(())
}

fn generate_round_trip_statement<D>(dialect: D, sql: &str) -> Statement
where
    D: Dialect,
{
    let statement = Parser::new(&dialect)
        .try_with_sql(sql)
        .unwrap()
        .parse_statement()
        .unwrap();

    let context = MockContextProvider {
        state: MockSessionState::default()
            .with_aggregate_function(sum_udaf())
            .with_aggregate_function(max_udaf())
            .with_aggregate_function(grouping_udaf())
            .with_window_function(rank_udwf())
            .with_scalar_function(Arc::new(unicode::substr().as_ref().clone()))
            .with_scalar_function(make_array_udf())
            .with_expr_planner(Arc::new(CoreFunctionPlanner::default()))
            .with_expr_planner(Arc::new(UnicodeFunctionPlanner))
            .with_expr_planner(Arc::new(NestedFunctionPlanner))
            .with_expr_planner(Arc::new(FieldAccessPlanner)),
    };
    let sql_to_rel = SqlToRel::new(&context);
    let plan = sql_to_rel.sql_statement_to_plan(statement).unwrap();

    plan_to_sql(&plan).unwrap()
}

#[test]
fn test_table_scan_alias() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Utf8, false),
    ]);

    let plan = table_scan(Some("t1"), &schema, None)?
        .project(vec![col("id")])?
        .alias("a")?
        .build()?;
    let sql = plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT * FROM (SELECT t1.id FROM t1) AS a"
    );

    let plan = table_scan(Some("t1"), &schema, None)?
        .project(vec![col("id")])?
        .alias("a")?
        .build()?;

    let sql = plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT * FROM (SELECT t1.id FROM t1) AS a"
    );

    let plan = table_scan(Some("t1"), &schema, None)?
        .filter(col("id").gt(lit(5)))?
        .project(vec![col("id")])?
        .alias("a")?
        .build()?;
    let sql = plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT * FROM (SELECT t1.id FROM t1 WHERE (t1.id > 5)) AS a"
    );

    let table_scan_with_two_filter = table_scan_with_filters(
        Some("t1"),
        &schema,
        None,
        vec![col("id").gt(lit(1)), col("age").lt(lit(2))],
    )?
    .project(vec![col("id")])?
    .alias("a")?
    .build()?;
    let table_scan_with_two_filter = plan_to_sql(&table_scan_with_two_filter)?;
    assert_snapshot!(
        table_scan_with_two_filter,
        @"SELECT a.id FROM t1 AS a WHERE ((a.id > 1) AND (a.age < 2))"
    );

    let table_scan_with_fetch =
        table_scan_with_filter_and_fetch(Some("t1"), &schema, None, vec![], Some(10))?
            .project(vec![col("id")])?
            .alias("a")?
            .build()?;
    let table_scan_with_fetch = plan_to_sql(&table_scan_with_fetch)?;
    assert_snapshot!(
        table_scan_with_fetch,
        @"SELECT a.id FROM (SELECT * FROM t1 LIMIT 10) AS a"
    );

    let table_scan_with_pushdown_all = table_scan_with_filter_and_fetch(
        Some("t1"),
        &schema,
        Some(vec![0, 1]),
        vec![col("id").gt(lit(1))],
        Some(10),
    )?
    .project(vec![col("id")])?
    .alias("a")?
    .build()?;
    let table_scan_with_pushdown_all = plan_to_sql(&table_scan_with_pushdown_all)?;
    assert_snapshot!(
        table_scan_with_pushdown_all,
        @"SELECT a.id FROM (SELECT a.id, a.age FROM t1 AS a WHERE (a.id > 1) LIMIT 10) AS a"
    );
    Ok(())
}

#[test]
fn test_table_scan_pushdown() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Utf8, false),
    ]);
    let scan_with_projection =
        table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let scan_with_projection = plan_to_sql(&scan_with_projection)?;
    assert_snapshot!(
        scan_with_projection,
        @"SELECT t1.id, t1.age FROM t1"
    );

    let scan_with_projection = table_scan(Some("t1"), &schema, Some(vec![1]))?.build()?;
    let scan_with_projection = plan_to_sql(&scan_with_projection)?;
    assert_snapshot!(
        scan_with_projection,
        @"SELECT t1.age FROM t1"
    );

    let scan_with_no_projection = table_scan(Some("t1"), &schema, None)?.build()?;
    let scan_with_no_projection = plan_to_sql(&scan_with_no_projection)?;
    assert_snapshot!(
        scan_with_no_projection,
        @"SELECT * FROM t1"
    );

    let table_scan_with_projection_alias =
        table_scan(Some("t1"), &schema, Some(vec![0, 1]))?
            .alias("ta")?
            .build()?;
    let table_scan_with_projection_alias =
        plan_to_sql(&table_scan_with_projection_alias)?;
    assert_snapshot!(
        table_scan_with_projection_alias,
        @"SELECT ta.id, ta.age FROM t1 AS ta"
    );

    let table_scan_with_projection_alias =
        table_scan(Some("t1"), &schema, Some(vec![1]))?
            .alias("ta")?
            .build()?;
    let table_scan_with_projection_alias =
        plan_to_sql(&table_scan_with_projection_alias)?;
    assert_snapshot!(
        table_scan_with_projection_alias,
        @"SELECT ta.age FROM t1 AS ta"
    );

    let table_scan_with_no_projection_alias = table_scan(Some("t1"), &schema, None)?
        .alias("ta")?
        .build()?;
    let table_scan_with_no_projection_alias =
        plan_to_sql(&table_scan_with_no_projection_alias)?;
    assert_snapshot!(
        table_scan_with_no_projection_alias,
        @"SELECT * FROM t1 AS ta"
    );

    let query_from_table_scan_with_projection = LogicalPlanBuilder::from(
        table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?,
    )
    .project(vec![col("id"), col("age")])?
    .build()?;
    let query_from_table_scan_with_projection =
        plan_to_sql(&query_from_table_scan_with_projection)?;
    assert_snapshot!(
        query_from_table_scan_with_projection,
        @"SELECT t1.id, t1.age FROM t1"
    );

    let query_from_table_scan_with_two_projections = LogicalPlanBuilder::from(
        table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?,
    )
    .project(vec![col("id"), col("age")])?
    .project(vec![wildcard()])?
    .build()?;
    let query_from_table_scan_with_two_projections =
        plan_to_sql(&query_from_table_scan_with_two_projections)?;
    assert_snapshot!(
        query_from_table_scan_with_two_projections,
        @"SELECT t1.id, t1.age FROM (SELECT t1.id, t1.age FROM t1)"
    );

    let table_scan_with_filter = table_scan_with_filters(
        Some("t1"),
        &schema,
        None,
        vec![col("id").gt(col("age"))],
    )?
    .build()?;
    let table_scan_with_filter = plan_to_sql(&table_scan_with_filter)?;
    assert_snapshot!(
        table_scan_with_filter,
        @"SELECT * FROM t1 WHERE (t1.id > t1.age)"
    );

    let table_scan_with_two_filter = table_scan_with_filters(
        Some("t1"),
        &schema,
        None,
        vec![col("id").gt(lit(1)), col("age").lt(lit(2))],
    )?
    .build()?;
    let table_scan_with_two_filter = plan_to_sql(&table_scan_with_two_filter)?;
    assert_snapshot!(
        table_scan_with_two_filter,
        @"SELECT * FROM t1 WHERE ((t1.id > 1) AND (t1.age < 2))"
    );

    let table_scan_with_filter_alias = table_scan_with_filters(
        Some("t1"),
        &schema,
        None,
        vec![col("id").gt(col("age"))],
    )?
    .alias("ta")?
    .build()?;
    let table_scan_with_filter_alias = plan_to_sql(&table_scan_with_filter_alias)?;
    assert_snapshot!(
        table_scan_with_filter_alias,
        @"SELECT * FROM t1 AS ta WHERE (ta.id > ta.age)"
    );

    let table_scan_with_projection_and_filter = table_scan_with_filters(
        Some("t1"),
        &schema,
        Some(vec![0, 1]),
        vec![col("id").gt(col("age"))],
    )?
    .build()?;
    let table_scan_with_projection_and_filter =
        plan_to_sql(&table_scan_with_projection_and_filter)?;
    assert_snapshot!(
        table_scan_with_projection_and_filter,
        @"SELECT t1.id, t1.age FROM t1 WHERE (t1.id > t1.age)"
    );

    let table_scan_with_projection_and_filter = table_scan_with_filters(
        Some("t1"),
        &schema,
        Some(vec![1]),
        vec![col("id").gt(col("age"))],
    )?
    .build()?;
    let table_scan_with_projection_and_filter =
        plan_to_sql(&table_scan_with_projection_and_filter)?;
    assert_snapshot!(
        table_scan_with_projection_and_filter,
        @"SELECT t1.age FROM t1 WHERE (t1.id > t1.age)"
    );

    let table_scan_with_inline_fetch =
        table_scan_with_filter_and_fetch(Some("t1"), &schema, None, vec![], Some(10))?
            .build()?;
    let table_scan_with_inline_fetch = plan_to_sql(&table_scan_with_inline_fetch)?;
    assert_snapshot!(
        table_scan_with_inline_fetch,
        @"SELECT * FROM t1 LIMIT 10"
    );

    let table_scan_with_projection_and_inline_fetch = table_scan_with_filter_and_fetch(
        Some("t1"),
        &schema,
        Some(vec![0, 1]),
        vec![],
        Some(10),
    )?
    .build()?;
    let table_scan_with_projection_and_inline_fetch =
        plan_to_sql(&table_scan_with_projection_and_inline_fetch)?;
    assert_snapshot!(
        table_scan_with_projection_and_inline_fetch,
        @"SELECT t1.id, t1.age FROM t1 LIMIT 10"
    );

    let table_scan_with_all = table_scan_with_filter_and_fetch(
        Some("t1"),
        &schema,
        Some(vec![0, 1]),
        vec![col("id").gt(col("age"))],
        Some(10),
    )?
    .build()?;
    let table_scan_with_all = plan_to_sql(&table_scan_with_all)?;
    assert_snapshot!(
        table_scan_with_all,
        @"SELECT t1.id, t1.age FROM t1 WHERE (t1.id > t1.age) LIMIT 10"
    );

    let table_scan_with_additional_filter = table_scan_with_filters(
        Some("t1"),
        &schema,
        None,
        vec![col("id").gt(col("age"))],
    )?
    .filter(col("id").eq(lit(5)))?
    .build()?;
    let table_scan_with_filter = plan_to_sql(&table_scan_with_additional_filter)?;
    assert_snapshot!(
        table_scan_with_filter,
        @"SELECT * FROM t1 WHERE (t1.id = 5) AND (t1.id > t1.age)"
    );

    Ok(())
}

#[test]
fn test_sort_with_push_down_fetch() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Utf8, false),
    ]);

    let plan = table_scan(Some("t1"), &schema, None)?
        .project(vec![col("id"), col("age")])?
        .sort_with_limit(vec![col("age").sort(true, true)], Some(10))?
        .build()?;

    let sql = plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT t1.id, t1.age FROM t1 ORDER BY t1.age ASC NULLS FIRST LIMIT 10"
    );
    Ok(())
}

#[test]
fn test_sort_with_scalar_fn_and_push_down_fetch() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("search_phrase", DataType::Utf8, false),
        Field::new("event_time", DataType::Utf8, false),
    ]);

    let substr_udf = unicode::substr();

    // Build a plan that mimics the DF52 optimizer output:
    // Projection(search_phrase) → Sort(substr(event_time), fetch=10)
    //   → Projection(search_phrase, event_time) → Filter → TableScan
    // This triggers a subquery because the outer projection differs from the inner one.
    // The ORDER BY scalar function must not reference the inner table qualifier.
    let plan = table_scan(Some("t1"), &schema, None)?
        .filter(col("search_phrase").not_eq(lit("")))?
        .project(vec![col("search_phrase"), col("event_time")])?
        .sort_with_limit(
            vec![
                substr_udf
                    .call(vec![col("event_time"), lit(1), lit(5)])
                    .sort(true, true),
            ],
            Some(10),
        )?
        .project(vec![col("search_phrase")])?
        .build()?;

    let sql = plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT t1.search_phrase FROM (SELECT t1.search_phrase, t1.event_time FROM t1 WHERE (t1.search_phrase <> '') ORDER BY substr(t1.event_time, 1, 5) ASC NULLS FIRST LIMIT 10)"
    );
    Ok(())
}

#[test]
fn test_join_with_table_scan_filters() -> Result<()> {
    let schema_left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let schema_right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Utf8, false),
    ]);

    let left_plan = table_scan_with_filters(
        Some("left_table"),
        &schema_left,
        None,
        vec![col("name").like(lit("some_name"))],
    )?
    .alias("left")?
    .build()?;

    let right_plan = table_scan_with_filters(
        Some("right_table"),
        &schema_right,
        None,
        vec![col("age").gt(lit(10))],
    )?
    .build()?;

    let join_plan_with_filter = LogicalPlanBuilder::from(left_plan.clone())
        .join(
            right_plan.clone(),
            datafusion_expr::JoinType::Inner,
            (vec!["left.id"], vec!["right_table.id"]),
            Some(col("left.id").gt(lit(5))),
        )?
        .build()?;

    let sql = plan_to_sql(&join_plan_with_filter)?;
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM left_table AS "left" INNER JOIN right_table ON "left".id = right_table.id AND ("left".id > 5) WHERE "left"."name" LIKE 'some_name' AND (age > 10)"#
    );

    let join_plan_no_filter = LogicalPlanBuilder::from(left_plan.clone())
        .join(
            right_plan,
            datafusion_expr::JoinType::Inner,
            (vec!["left.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;

    let sql = plan_to_sql(&join_plan_no_filter)?;
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM left_table AS "left" INNER JOIN right_table ON "left".id = right_table.id WHERE "left"."name" LIKE 'some_name' AND (age > 10)"#
    );

    let right_plan_with_filter = table_scan_with_filters(
        Some("right_table"),
        &schema_right,
        None,
        vec![col("age").gt(lit(10))],
    )?
    .filter(col("right_table.name").eq(lit("before_join_filter_val")))?
    .build()?;

    let join_plan_multiple_filters = LogicalPlanBuilder::from(left_plan.clone())
        .join(
            right_plan_with_filter,
            datafusion_expr::JoinType::Inner,
            (vec!["left.id"], vec!["right_table.id"]),
            Some(col("left.id").gt(lit(5))),
        )?
        .filter(col("left.name").eq(lit("after_join_filter_val")))?
        .build()?;

    let sql = plan_to_sql(&join_plan_multiple_filters)?;
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM left_table AS "left" INNER JOIN right_table ON "left".id = right_table.id AND ("left".id > 5) WHERE ("left"."name" = 'after_join_filter_val') AND "left"."name" LIKE 'some_name' AND (right_table."name" = 'before_join_filter_val') AND (age > 10)"#
    );

    let right_plan_with_filter_schema = table_scan_with_filters(
        Some("right_table"),
        &schema_right,
        None,
        vec![
            col("right_table.age").gt(lit(10)),
            col("right_table.age").lt(lit(11)),
        ],
    )?
    .build()?;
    let right_plan_with_duplicated_filter =
        LogicalPlanBuilder::from(right_plan_with_filter_schema.clone())
            .filter(col("right_table.age").gt(lit(10)))?
            .build()?;

    let join_plan_duplicated_filter = LogicalPlanBuilder::from(left_plan)
        .join(
            right_plan_with_duplicated_filter,
            datafusion_expr::JoinType::Inner,
            (vec!["left.id"], vec!["right_table.id"]),
            Some(col("left.id").gt(lit(5))),
        )?
        .build()?;

    let sql = plan_to_sql(&join_plan_duplicated_filter)?;
    assert_snapshot!(
        sql,
        @r#"SELECT * FROM left_table AS "left" INNER JOIN right_table ON "left".id = right_table.id AND ("left".id > 5) WHERE "left"."name" LIKE 'some_name' AND (right_table.age > 10) AND (right_table.age < 11)"#
    );

    // Inner join with a scalar subquery in table_scan_filters. The subquery filter should appear in WHERE, not in JOIN ON,
    // since dialects like BigQuery reject subqueries in join predicates.
    let schema_subquery = Schema::new(vec![Field::new("id", DataType::Utf8, false)]);
    let subquery_plan = table_scan(Some("subquery_table"), &schema_subquery, None)?
        .aggregate(vec![] as Vec<Expr>, vec![max(col("subquery_table.id"))])?
        .build()?;
    let right_plan_with_subquery = table_scan_with_filters(
        Some("right_table"),
        &schema_right,
        None,
        vec![col("right_table.id").eq(scalar_subquery(Arc::new(subquery_plan)))],
    )?
    .build()?;

    let left_plan =
        table_scan(Some("left_table"), &schema_left, Some(vec![0, 1]))?.build()?;

    let join_plan_subquery_filter = LogicalPlanBuilder::from(left_plan)
        .join(
            right_plan_with_subquery,
            datafusion_expr::JoinType::Inner,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;

    let sql = plan_to_sql(&join_plan_subquery_filter)?;
    assert_snapshot!(
        sql,
        @r#"SELECT left_table.id, left_table."name" FROM left_table INNER JOIN right_table ON left_table.id = right_table.id WHERE (right_table.id = (SELECT max(subquery_table.id) FROM subquery_table))"#
    );

    // Inner join with an IN subquery in table_scan_filters.
    let subquery_plan_in = table_scan(Some("subquery_table"), &schema_subquery, None)?
        .project(vec![col("subquery_table.id")])?
        .build()?;
    let right_plan_with_in = table_scan_with_filters(
        Some("right_table"),
        &schema_right,
        None,
        vec![in_subquery(
            col("right_table.id"),
            Arc::new(subquery_plan_in),
        )],
    )?
    .build()?;

    let left_plan_in =
        table_scan(Some("left_table"), &schema_left, Some(vec![0, 1]))?.build()?;

    let join_plan_in_subquery = LogicalPlanBuilder::from(left_plan_in)
        .join(
            right_plan_with_in,
            datafusion_expr::JoinType::Inner,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;

    let sql = plan_to_sql(&join_plan_in_subquery)?;
    assert_snapshot!(
        sql,
        @r#"SELECT left_table.id, left_table."name" FROM left_table INNER JOIN right_table ON left_table.id = right_table.id WHERE right_table.id IN (SELECT subquery_table.id FROM subquery_table)"#
    );

    // Inner join with an EXISTS subquery in table_scan_filters.
    let subquery_plan_exists =
        table_scan(Some("subquery_table"), &schema_subquery, None)?
            .filter(col("subquery_table.id").eq(col("right_table.id")))?
            .build()?;
    let right_plan_with_exists = table_scan_with_filters(
        Some("right_table"),
        &schema_right,
        None,
        vec![exists(Arc::new(subquery_plan_exists))],
    )?
    .build()?;

    let left_plan_exists =
        table_scan(Some("left_table"), &schema_left, Some(vec![0, 1]))?.build()?;

    let join_plan_exists = LogicalPlanBuilder::from(left_plan_exists)
        .join(
            right_plan_with_exists,
            datafusion_expr::JoinType::Inner,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;

    let sql = plan_to_sql(&join_plan_exists)?;
    assert_snapshot!(
        sql,
        @r#"SELECT left_table.id, left_table."name" FROM left_table INNER JOIN right_table ON left_table.id = right_table.id WHERE EXISTS (SELECT * FROM subquery_table WHERE (subquery_table.id = right_table.id))"#
    );

    Ok(())
}

#[test]
fn test_outer_join_with_table_scan_filters() -> Result<()> {
    let schema_left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let schema_right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let left_with_filter = || {
        table_scan_with_filters(
            Some("left_table"),
            &schema_left,
            Some(vec![0, 1]),
            vec![col("left_table.id").eq(lit("a"))],
        )?
        .build()
    };
    let right_with_filter = || {
        table_scan_with_filters(
            Some("right_table"),
            &schema_right,
            Some(vec![0, 1]),
            vec![col("right_table.age").gt(lit(10))],
        )?
        .build()
    };
    let plain_left =
        || table_scan(Some("left_table"), &schema_left, Some(vec![0, 1]))?.build();
    let plain_right =
        || table_scan(Some("right_table"), &schema_right, Some(vec![0, 1]))?.build();

    // LEFT JOIN, filter on the preserved (left) side: it must land in WHERE.
    // Folding it into `ON` would not remove a single row, because a LEFT JOIN
    // preserves every left row regardless of the `ON` predicate.
    let plan = LogicalPlanBuilder::from(left_with_filter()?)
        .join(
            plain_right()?,
            datafusion_expr::JoinType::Left,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table LEFT OUTER JOIN right_table ON left_table.id = right_table.id WHERE (left_table.id = 'a')"#
    );

    // LEFT JOIN, filter on the non-preserved (right) side: it must stay in
    // `ON`. Moving it to WHERE would discard the null-extended rows and turn
    // the LEFT JOIN into an INNER JOIN.
    let plan = LogicalPlanBuilder::from(plain_left()?)
        .join(
            right_with_filter()?,
            datafusion_expr::JoinType::Left,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table LEFT OUTER JOIN right_table ON left_table.id = right_table.id AND (right_table.age > 10)"#
    );

    // LEFT JOIN with a filter on each side: the two are routed to different
    // clauses, and neither is dropped.
    let plan = LogicalPlanBuilder::from(left_with_filter()?)
        .join(
            right_with_filter()?,
            datafusion_expr::JoinType::Left,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table LEFT OUTER JOIN right_table ON left_table.id = right_table.id AND (right_table.age > 10) WHERE (left_table.id = 'a')"#
    );

    // RIGHT JOIN is the mirror image: the right side is preserved.
    let plan = LogicalPlanBuilder::from(left_with_filter()?)
        .join(
            right_with_filter()?,
            datafusion_expr::JoinType::Right,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table RIGHT OUTER JOIN right_table ON left_table.id = right_table.id AND (left_table.id = 'a') WHERE (right_table.age > 10)"#
    );

    // FULL OUTER JOIN preserves both sides, so neither `ON` nor `WHERE`
    // expresses an input filter correctly. Isolate the filtered input in a
    // derived table so rows rejected by the filter cannot reappear as unmatched
    // rows from the FULL JOIN.
    let plan = LogicalPlanBuilder::from(left_with_filter()?)
        .join(
            plain_right()?,
            datafusion_expr::JoinType::Full,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM (SELECT left_table.id, left_table."name" FROM left_table WHERE (left_table.id = 'a')) AS left_table FULL JOIN right_table ON left_table.id = right_table.id"#
    );

    // The aliased form: the filter is rewritten to the alias on its way out of
    // the scan, and must still land in WHERE.
    let aliased_left = table_scan_with_filters(
        Some("left_table"),
        &schema_left,
        Some(vec![0, 1]),
        vec![col("left_table.id").eq(lit("a"))],
    )?
    .alias("l")?
    .build()?;
    let plan = LogicalPlanBuilder::from(aliased_left)
        .join(
            plain_right()?,
            datafusion_expr::JoinType::Left,
            (vec!["l.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT l.id, l."name", right_table.id, right_table.age FROM left_table AS l LEFT OUTER JOIN right_table ON l.id = right_table.id WHERE (l.id = 'a')"#
    );

    Ok(())
}

#[test]
fn test_interval_lhs_eq() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        "select interval '2 seconds' = interval '2 seconds'",
    );
    assert_snapshot!(
        statement,
        @"SELECT (INTERVAL '2.000000000 SECS' = INTERVAL '2.000000000 SECS')"
    )
}

#[test]
fn test_interval_lhs_lt() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        "select interval '2 seconds' < interval '2 seconds'",
    );
    assert_snapshot!(
        statement,
        @"SELECT (INTERVAL '2.000000000 SECS' < INTERVAL '2.000000000 SECS')"
    )
}

#[test]
fn test_without_offset() {
    let statement = generate_round_trip_statement(MySqlDialect {}, "select 1");
    assert_snapshot!(
        statement,
        @"SELECT 1"
    )
}

#[test]
fn test_cast_to_tinyint() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select cast(3 as tinyint)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserPostgreSqlDialect {},
        expected: @"SELECT CAST(3 AS SMALLINT)",
    );
    Ok(())
}

#[test]
fn test_cast_to_tinyint_default_dialect() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "select cast(3 as tinyint)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT CAST(3 AS TINYINT)",
    );
    Ok(())
}

#[test]
fn test_with_offset0() {
    let statement = generate_round_trip_statement(MySqlDialect {}, "select 1 offset 0");
    assert_snapshot!(
        statement,
        @"SELECT 1 OFFSET 0"
    )
}

#[test]
fn test_with_offset95() {
    let statement = generate_round_trip_statement(MySqlDialect {}, "select 1 offset 95");
    assert_snapshot!(
        statement,
        @"SELECT 1 OFFSET 95"
    )
}

#[test]
fn test_order_by_to_sql_1() {
    // order by aggregation function
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT id, first_name, SUM(id) FROM person GROUP BY id, first_name ORDER BY SUM(id) ASC, first_name DESC, id, first_name LIMIT 10"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.id, person.first_name, sum(person.id) FROM person GROUP BY person.id, person.first_name ORDER BY sum(person.id) ASC NULLS LAST, person.first_name DESC NULLS FIRST, person.id ASC NULLS LAST, person.first_name ASC NULLS LAST LIMIT 10"
    );
}

#[test]
fn test_order_by_to_sql_2() {
    // order by aggregation function alias
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT id, first_name, SUM(id) as total_sum FROM person GROUP BY id, first_name ORDER BY total_sum ASC, first_name DESC, id, first_name LIMIT 10"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.id, person.first_name, sum(person.id) AS total_sum FROM person GROUP BY person.id, person.first_name ORDER BY total_sum ASC NULLS LAST, person.first_name DESC NULLS FIRST, person.id ASC NULLS LAST, person.first_name ASC NULLS LAST LIMIT 10"
    );
}

#[test]
fn test_order_by_to_sql_3() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT id, first_name, substr(first_name,0,5) FROM person ORDER BY id, substr(first_name,0,5)"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.id, person.first_name, substr(person.first_name, 0, 5) FROM person ORDER BY person.id ASC NULLS LAST, substr(person.first_name, 0, 5) ASC NULLS LAST"
    );
}

#[test]
fn test_complex_order_by_with_grouping() -> Result<()> {
    let state = MockSessionState::default().with_aggregate_function(grouping_udaf());

    let context = MockContextProvider { state };
    let sql_to_rel = SqlToRel::new(&context);

    // This SQL is based on a simplified version of the TPC-DS query 36.
    let statement = Parser::new(&GenericDialect {})
        .try_with_sql(
            r#"SELECT
            j1_id,
            j1_string,
            grouping(j1_id) + grouping(j1_string) as lochierarchy
        FROM
            j1
        GROUP BY
            ROLLUP (j1_id, j1_string)
        ORDER BY
            grouping(j1_id) + grouping(j1_string) DESC,
            CASE
                WHEN grouping(j1_id) + grouping(j1_string) = 0 THEN j1_id
            END
        LIMIT 100"#,
        )?
        .parse_statement()?;

    let plan = sql_to_rel.sql_statement_to_plan(statement)?;
    let unparser = Unparser::default();
    let sql = unparser.plan_to_sql(&plan)?;
    insta::with_settings!({
        filters => vec![
            // Force a deterministic order for the grouping pairs
            (r#"grouping\(j1\.(?:j1_id|j1_string)\),\s*grouping\(j1\.(?:j1_id|j1_string)\)"#, "grouping(j1.j1_string), grouping(j1.j1_id)")
        ],
    }, {
        assert_snapshot!(
            sql,
            @"SELECT j1.j1_id, j1.j1_string, (grouping(j1.j1_id) + grouping(j1.j1_string)) AS lochierarchy FROM j1 GROUP BY ROLLUP (j1.j1_id, j1.j1_string) ORDER BY lochierarchy DESC NULLS FIRST, CASE WHEN ((grouping(j1.j1_id) + grouping(j1.j1_string)) = 0) THEN j1.j1_id END ASC NULLS LAST LIMIT 100"
        );
    });

    Ok(())
}

/// Regression test: a computed (`BinaryExpr`) SELECT-list alias referenced
/// inside an `ORDER BY` *expression* (e.g. `CASE WHEN`) must be unparsed as the
/// underlying expression, not left as a bare alias column.
///
/// `unproject_sort_expr` previously only re-inlined `ScalarFunction` projection
/// expressions; any other computed expression (here `id + age`) fell through and
/// the alias leaked into the `ORDER BY`. This produces SQL that is invalid under
/// the SQL standard's name-resolution rules for `ORDER BY`: a bare output-column
/// alias is accepted as a top-level sort key, but when the alias appears inside a
/// larger expression, identifiers are resolved against the `FROM` relations only —
/// where the alias does not exist.
///
/// So the buggy output
/// `... ORDER BY CASE WHEN (computed = 0) THEN t1.id END`
/// is rejected by every standard-conforming engine, e.g.:
///   - PostgreSQL: `ERROR: column "computed" does not exist` (SQLSTATE 42703)
///   - DuckDB:     `Binder Error: Referenced column "computed" not found`
///   - MySQL:      `ERROR 1054 (42S22): Unknown column 'computed' in 'order clause'`
///
/// This was the root cause of the `column "lochierarchy" does not exist` failures
/// observed in federated TPC-DS queries (q36 / q70 / q86), where `lochierarchy` is
/// a `grouping(a) + grouping(b)` alias used inside an `ORDER BY CASE WHEN`.
///
/// ```text
///   t1
///  ┌─────┬─────┐    ```sql
///  │ id  │ i32 │      SELECT id + age AS computed, id
///  ├─────┼─────┤      FROM t1
///  │ age │ i32 │      ORDER BY CASE WHEN computed = 0 THEN id END
///  └─────┴─────┘    ```
///
///  ── logical plan ───────────────────────────────────────────────────────────
///   Sort: CASE WHEN computed = 0 THEN id END ASC NULLS FIRST
///     Projection: (id + age) AS computed, id
///       TableScan: t1
///
///  ── unparsed SQL, before the fix (BROKEN) ──────────────────────────────────
///   SELECT (t1.id + t1.age) AS computed, t1.id
///   FROM t1
///   ORDER BY CASE WHEN (computed = 0) THEN t1.id END
///                      ▲
///                      └─ `computed` is a SELECT alias, not a column of t1;
///                         inside an expression the engine resolves it against
///                         FROM only → PostgreSQL: ERROR: column "computed"
///                         does not exist (SQLSTATE 42703)
///
///  ── Unparsed SQL, after the fix (VALID) ────────────────────────────────────
///   SELECT (t1.id + t1.age) AS computed, t1.id
///   FROM t1
///   ORDER BY CASE WHEN ((t1.id + t1.age) = 0) THEN t1.id END
///                       ▲
///                       └─ underlying expression inlined; resolves against t1
///                          → valid on every standard-conforming engine
/// ```
#[test]
fn test_order_by_with_computed_alias_inside_expr() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("age", DataType::Int32, false),
    ]);

    // Build plan:  Sort(CASE WHEN computed = 0 THEN id END)
    //                Projection((id + age) AS computed, id)
    //                  TableScan(t1)
    //
    // `computed` is a BinaryExpr alias in the Projection.  The sort expression
    // references it via col("computed") inside a CASE.  Without the fix the
    // unparser emits `CASE WHEN (computed = 0) THEN t1.id END` which is invalid
    // SQL in PostgreSQL; with the fix it emits
    // `CASE WHEN ((t1.id + t1.age) = 0) THEN t1.id END`.
    let computed_alias = (col("id") + col("age")).alias("computed");
    let case_expr = datafusion_expr::when(col("computed").eq(lit(0i32)), col("id"))
        .end()
        .unwrap();

    let plan = table_scan(Some("t1"), &schema, None)?
        .project(vec![computed_alias, col("id")])?
        .sort(vec![case_expr.sort(true, true)])?
        .build()?;

    let sql = plan_to_sql(&plan)?;
    // The CASE condition must reference the inlined expression, not the bare alias.
    assert_snapshot!(
        sql,
        @"SELECT (t1.id + t1.age) AS computed, t1.id FROM t1 ORDER BY CASE WHEN ((t1.id + t1.age) = 0) THEN t1.id END ASC NULLS FIRST"
    );
    Ok(())
}

/// Regression test: a sort key that is exactly a SELECT-list alias (here for a
/// BinaryExpr referencing a window function output) must be emitted as the bare
/// alias, not inlined.
///
/// Pattern from TPC-DS q12/q20/q98:
///   SELECT ..., sum(ws) * 100 / sum(sum(ws)) OVER (PARTITION BY i_class ...) AS revenueratio
///   FROM ...
///   ORDER BY revenueratio
///
/// A bare output-column alias is a valid top-level ORDER BY key in every
/// dialect, and inlining the aliased expression is harmful: engines that
/// re-plan and re-unparse the received SQL (e.g. a federated remote) can leak
/// the window's internal schema name as a quoted identifier, e.g.
///   ((sum(ws) * 100) / "sum(sum(ws_ext_sales_price)) PARTITION BY [i_class] ...")
/// which the downstream engine rejects (PostgreSQL 42703 / DuckDB Binder Error).
#[test]
fn test_order_by_with_binary_expr_referencing_window_output() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("class", DataType::Utf8, false),
        Field::new("sales_price", DataType::Float64, true),
    ]);

    // GROUP BY class, sum(sales_price) — left unaliased so the window's argument
    // is the raw aggregate-output column (as in q12), which the ORDER BY
    // unprojection must resolve back to the aggregate function call.
    let plan = table_scan(Some("t"), &schema, None)?
        .aggregate(vec![col("class")], vec![sum(col("sales_price"))])?
        .build()?;
    let agg_col = plan.schema().fields().last().unwrap().name().clone();

    // Build: sum(<agg output>) OVER (PARTITION BY class)
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::AggregateUDF(sum_udaf()),
        params: WindowFunctionParams {
            args: vec![col(agg_col.clone())],
            partition_by: vec![col("class")],
            order_by: vec![],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }));
    // Get the auto-generated window output column name from the schema
    let plan = LogicalPlanBuilder::from(plan)
        .window(vec![window_expr])?
        .build()?;
    let window_col_name = plan.schema().fields().last().unwrap().name().clone();

    // revenueratio = sum(sales_price) * 100 / (sum(sum(sales_price)) OVER (...))
    let revenueratio = ((col(agg_col.clone()) * lit(100i64))
        / col(window_col_name.clone()))
    .alias("revenueratio");

    let plan = LogicalPlanBuilder::from(plan)
        .project(vec![col("class"), revenueratio])?
        .sort(vec![col("revenueratio").sort(true, true)])?
        .build()?;

    let sql_str = plan_to_sql(&plan)?.to_string();
    // The sort key is exactly the projection alias, so ORDER BY must keep the
    // bare alias rather than inlining the window/aggregate expression.
    assert!(
        sql_str.ends_with("ORDER BY revenueratio ASC NULLS FIRST"),
        "ORDER BY should be the bare output-column alias, got: {sql_str}"
    );
    // Neither the window output column nor its nested aggregate argument may
    // appear anywhere as a quoted DataFusion internal identifier (both would
    // be rejected by PostgreSQL as 42703).
    assert!(
        !sql_str.contains(&format!("\"{window_col_name}\"")),
        "leaked the window output identifier, got: {sql_str}"
    );
    assert!(
        !sql_str.contains(&format!("\"{agg_col}\"")),
        "leaked the nested aggregate identifier, got: {sql_str}"
    );
    Ok(())
}

/// Companion to [`test_order_by_with_binary_expr_referencing_window_output`]:
/// when the same window-referencing alias appears *nested* inside a larger
/// ORDER BY expression, keeping the bare alias is not an option (engines
/// resolve identifiers inside expressions against the FROM relations only), so
/// the unparser must inline the aliased expression — resolving the window
/// output column to its full OVER expression and the aggregate-output column
/// nested in the window's argument to the aggregate function call, leaving
/// neither as a quoted DataFusion internal identifier.
#[test]
fn test_order_by_window_output_alias_nested_inside_expr() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("class", DataType::Utf8, false),
        Field::new("sales_price", DataType::Float64, true),
    ]);

    let plan = table_scan(Some("t"), &schema, None)?
        .aggregate(vec![col("class")], vec![sum(col("sales_price"))])?
        .build()?;
    let agg_col = plan.schema().fields().last().unwrap().name().clone();

    // Build: sum(<agg output>) OVER (PARTITION BY class)
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::AggregateUDF(sum_udaf()),
        params: WindowFunctionParams {
            args: vec![col(agg_col.clone())],
            partition_by: vec![col("class")],
            order_by: vec![],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }));
    let plan = LogicalPlanBuilder::from(plan)
        .window(vec![window_expr])?
        .build()?;
    let window_col_name = plan.schema().fields().last().unwrap().name().clone();

    let revenueratio = ((col(agg_col.clone()) * lit(100i64))
        / col(window_col_name.clone()))
    .alias("revenueratio");

    // Sort key: CASE WHEN revenueratio = 0 THEN class END — the alias is
    // nested inside a larger expression, so it must be inlined.
    let case_expr =
        datafusion_expr::when(col("revenueratio").eq(lit(0i64)), col("class"))
            .end()
            .unwrap();

    let plan = LogicalPlanBuilder::from(plan)
        .project(vec![col("class"), revenueratio])?
        .sort(vec![case_expr.sort(true, true)])?
        .build()?;

    let sql_str = plan_to_sql(&plan)?.to_string();
    // Scope the ORDER BY assertions to the clause itself: the SELECT list also
    // inlines the window expression, so whole-string checks for OVER / the
    // alias would pass regardless of what the ORDER BY contains.
    let order_by = sql_str
        .split_once("ORDER BY")
        .expect("SQL should contain ORDER BY")
        .1;
    // The sort key must inline the window function expression...
    assert!(
        order_by.to_uppercase().contains("OVER"),
        "ORDER BY should contain an inlined OVER clause, got: {sql_str}"
    );
    // ...with no reference to the SELECT-list alias (engines resolve
    // identifiers inside ORDER BY expressions against FROM relations only)...
    assert!(
        !order_by.contains("revenueratio"),
        "ORDER BY leaked the bare alias inside an expression, got: {sql_str}"
    );
    // ...and no internal identifiers anywhere.
    assert!(
        !sql_str.contains(&format!("\"{window_col_name}\"")),
        "leaked the window output identifier, got: {sql_str}"
    );
    assert!(
        !sql_str.contains(&format!("\"{agg_col}\"")),
        "leaked the nested aggregate identifier, got: {sql_str}"
    );
    Ok(())
}

#[test]
fn test_aggregation_to_sql() {
    let sql = r#"SELECT id, first_name,
        SUM(id) AS total_sum,
        SUM(id) OVER (PARTITION BY first_name ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING) AS moving_sum,
        SUM(id) FILTER (WHERE id > 50 AND first_name = 'John') OVER (PARTITION BY first_name ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING) AS filtered_sum,
        MAX(SUM(id)) OVER (PARTITION BY first_name ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS max_total,
        rank() OVER (PARTITION BY grouping(id) + grouping(age), CASE WHEN grouping(age) = 0 THEN id END ORDER BY sum(id) DESC) AS rank_within_parent_1,
        rank() OVER (PARTITION BY grouping(age) + grouping(id), CASE WHEN (CAST(grouping(age) AS BIGINT) = 0) THEN id END ORDER BY sum(id) DESC) AS rank_within_parent_2
        FROM person
        GROUP BY id, first_name"#;
    let statement = generate_round_trip_statement(GenericDialect {}, sql);
    assert_snapshot!(
        statement,
        @"SELECT person.id, person.first_name, sum(person.id) AS total_sum, sum(person.id) OVER (PARTITION BY person.first_name ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING) AS moving_sum, sum(person.id) FILTER (WHERE ((person.id > 50) AND (person.first_name = 'John'))) OVER (PARTITION BY person.first_name ROWS BETWEEN 5 PRECEDING AND 2 FOLLOWING) AS filtered_sum, max(sum(person.id)) OVER (PARTITION BY person.first_name ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS max_total, rank() OVER (PARTITION BY (grouping(person.id) + grouping(person.age)), CASE WHEN (grouping(person.age) = 0) THEN person.id END ORDER BY sum(person.id) DESC NULLS FIRST RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rank_within_parent_1, rank() OVER (PARTITION BY (grouping(person.age) + grouping(person.id)), CASE WHEN (CAST(grouping(person.age) AS BIGINT) = 0) THEN person.id END ORDER BY sum(person.id) DESC NULLS FIRST RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rank_within_parent_2 FROM person GROUP BY person.id, person.first_name",
    );
}

#[test]
fn test_unnest_to_sql_1() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT unnest(array_col) as u1, struct_col, array_col FROM unnest_table WHERE array_col != NULL ORDER BY struct_col, array_col"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT UNNEST(unnest_table.array_col) AS u1, unnest_table.struct_col, unnest_table.array_col FROM unnest_table WHERE (unnest_table.array_col <> NULL) ORDER BY unnest_table.struct_col ASC NULLS LAST, unnest_table.array_col ASC NULLS LAST"
    );
}

#[test]
fn test_unnest_to_sql_2() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT unnest(make_array(1, 2, 2, 5, NULL)) as u1"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT UNNEST([1, 2, 2, 5, NULL]) AS u1"
    );
}

#[test]
fn test_join_with_no_conditions() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        "SELECT j1.j1_id, j1.j1_string FROM j1 CROSS JOIN j2",
    );
    assert_snapshot!(
        statement,
        @"SELECT j1.j1_id, j1.j1_string FROM j1 CROSS JOIN j2"
    );
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd)]
struct MockUserDefinedLogicalPlan {
    input: LogicalPlan,
}

impl UserDefinedLogicalNodeCore for MockUserDefinedLogicalPlan {
    fn name(&self) -> &str {
        "MockUserDefinedLogicalPlan"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MockUserDefinedLogicalPlan")
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        Ok(MockUserDefinedLogicalPlan {
            input: inputs.into_iter().next().unwrap(),
        })
    }
}

struct MockStatementUnparser {}

impl UserDefinedLogicalNodeUnparser for MockStatementUnparser {
    fn unparse_to_statement(
        &self,
        node: &dyn UserDefinedLogicalNode,
        unparser: &Unparser,
    ) -> Result<UnparseToStatementResult> {
        if let Some(plan) = node.as_any().downcast_ref::<MockUserDefinedLogicalPlan>() {
            let input = unparser.plan_to_sql(&plan.input)?;
            Ok(UnparseToStatementResult::Modified(input))
        } else {
            Ok(UnparseToStatementResult::Unmodified)
        }
    }
}

struct UnusedUnparser {}

impl UserDefinedLogicalNodeUnparser for UnusedUnparser {
    fn unparse(
        &self,
        _node: &dyn UserDefinedLogicalNode,
        _unparser: &Unparser,
        _query: &mut Option<&mut QueryBuilder>,
        _select: &mut Option<&mut SelectBuilder>,
        _relation: &mut Option<&mut RelationBuilder>,
    ) -> Result<UnparseWithinStatementResult> {
        panic!("This should not be called");
    }

    fn unparse_to_statement(
        &self,
        _node: &dyn UserDefinedLogicalNode,
        _unparser: &Unparser,
    ) -> Result<UnparseToStatementResult> {
        panic!("This should not be called");
    }
}

#[test]
fn test_unparse_extension_to_statement() -> Result<()> {
    let dialect = GenericDialect {};
    let statement = Parser::new(&dialect)
        .try_with_sql("SELECT * FROM j1")?
        .parse_statement()?;
    let state = MockSessionState::default();
    let context = MockContextProvider { state };
    let sql_to_rel = SqlToRel::new(&context);
    let plan = sql_to_rel.sql_statement_to_plan(statement)?;

    let extension = MockUserDefinedLogicalPlan { input: plan };
    let extension = LogicalPlan::Extension(Extension {
        node: Arc::new(extension),
    });
    let unparser = Unparser::default().with_extension_unparsers(vec![
        Arc::new(MockStatementUnparser {}),
        Arc::new(UnusedUnparser {}),
    ]);
    let sql = unparser.plan_to_sql(&extension)?;
    assert_snapshot!(
        sql,
        @"SELECT j1.j1_id, j1.j1_string FROM j1"
    );

    if let Some(err) = plan_to_sql(&extension).err() {
        assert_contains!(
            err.to_string(),
            "This feature is not implemented: Unsupported extension node: MockUserDefinedLogicalPlan"
        );
    } else {
        panic!("Expected error");
    }
    Ok(())
}

struct MockSqlUnparser {}

impl UserDefinedLogicalNodeUnparser for MockSqlUnparser {
    fn unparse(
        &self,
        node: &dyn UserDefinedLogicalNode,
        unparser: &Unparser,
        _query: &mut Option<&mut QueryBuilder>,
        _select: &mut Option<&mut SelectBuilder>,
        relation: &mut Option<&mut RelationBuilder>,
    ) -> Result<UnparseWithinStatementResult> {
        if let Some(plan) = node.as_any().downcast_ref::<MockUserDefinedLogicalPlan>() {
            let Statement::Query(input) = unparser.plan_to_sql(&plan.input)? else {
                return Ok(UnparseWithinStatementResult::Unmodified);
            };
            let mut derived_builder = DerivedRelationBuilder::default();
            derived_builder.subquery(input);
            derived_builder.lateral(false);
            if let Some(rel) = relation {
                rel.derived(derived_builder);
            }
        }
        Ok(UnparseWithinStatementResult::Modified)
    }
}

#[test]
fn test_unparse_extension_to_sql() -> Result<()> {
    let dialect = GenericDialect {};
    let statement = Parser::new(&dialect)
        .try_with_sql("SELECT * FROM j1")?
        .parse_statement()?;
    let state = MockSessionState::default();
    let context = MockContextProvider { state };
    let sql_to_rel = SqlToRel::new(&context);
    let plan = sql_to_rel.sql_statement_to_plan(statement)?;

    let extension = MockUserDefinedLogicalPlan { input: plan };
    let extension = LogicalPlan::Extension(Extension {
        node: Arc::new(extension),
    });

    let plan = LogicalPlanBuilder::from(extension)
        .project(vec![col("j1_id").alias("user_id")])?
        .build()?;
    let unparser = Unparser::default().with_extension_unparsers(vec![
        Arc::new(MockSqlUnparser {}),
        Arc::new(UnusedUnparser {}),
    ]);
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT j1.j1_id AS user_id FROM (SELECT j1.j1_id, j1.j1_string FROM j1)"
    );

    if let Some(err) = plan_to_sql(&plan).err() {
        assert_contains!(
            err.to_string(),
            "This feature is not implemented: Unsupported extension node: MockUserDefinedLogicalPlan"
        );
    } else {
        panic!("Expected error")
    }
    Ok(())
}

#[test]
fn test_unparse_optimized_multi_union() -> Result<()> {
    let unparser = Unparser::default();

    let schema = Schema::new(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Utf8, false),
    ]);

    let dfschema = Arc::new(DFSchema::try_from(schema)?);

    let empty = LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: true,
        schema: dfschema.clone(),
    });

    let plan = LogicalPlan::Union(Union {
        inputs: vec![
            project(empty.clone(), vec![lit(1).alias("x"), lit("a").alias("y")])?.into(),
            project(empty.clone(), vec![lit(1).alias("x"), lit("b").alias("y")])?.into(),
            project(empty.clone(), vec![lit(2).alias("x"), lit("a").alias("y")])?.into(),
            project(empty.clone(), vec![lit(2).alias("x"), lit("c").alias("y")])?.into(),
        ],
        schema: dfschema.clone(),
    });
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @"SELECT 1 AS x, 'a' AS y UNION ALL SELECT 1 AS x, 'b' AS y UNION ALL SELECT 2 AS x, 'a' AS y UNION ALL SELECT 2 AS x, 'c' AS y"
    );

    let plan = LogicalPlan::Union(Union {
        inputs: vec![
            project(empty.clone(), vec![lit(1).alias("x"), lit("a").alias("y")])?.into(),
        ],
        schema: dfschema.clone(),
    });

    if let Some(err) = plan_to_sql(&plan).err() {
        assert_contains!(err.to_string(), "UNION operator requires at least 2 inputs");
    } else {
        panic!("Expected error")
    }

    Ok(())
}

/// Test unparse the optimized plan from the following SQL:
/// ```
/// SELECT
///   customer_view.c_custkey,
///   customer_view.c_name,
///   customer_view.custkey_plus
/// FROM
///   (
///     SELECT
///       customer.c_custkey,
///       customer.c_name,
///       customer.custkey_plus
///     FROM
///       (
///         SELECT
///           customer.c_custkey,
///           CAST(customer.c_custkey AS BIGINT) + 1 AS custkey_plus,
///           customer.c_name
///         FROM
///           (
///             SELECT
///               customer.c_custkey AS c_custkey,
///               customer.c_name AS c_name
///             FROM
///               customer
///           ) AS customer
///       ) AS customer
///   ) AS customer_view
/// ```
#[test]
fn test_unparse_subquery_alias_with_table_pushdown() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_name", DataType::Utf8, false),
    ]);

    let table_scan = table_scan(Some("customer"), &schema, Some(vec![0, 1]))?.build()?;

    let plan = LogicalPlanBuilder::from(table_scan)
        .alias("customer")?
        .project(vec![
            col("customer.c_custkey"),
            cast(col("customer.c_custkey"), DataType::Int64)
                .add(lit(1))
                .alias("custkey_plus"),
            col("customer.c_name"),
        ])?
        .alias("customer")?
        .project(vec![
            col("customer.c_custkey"),
            col("customer.c_name"),
            col("customer.custkey_plus"),
        ])?
        .alias("customer_view")?
        .build()?;

    let unparser = Unparser::default();
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT customer_view.c_custkey, customer_view.c_name, customer_view.custkey_plus FROM (SELECT customer.c_custkey, (CAST(customer.c_custkey AS BIGINT) + 1) AS custkey_plus, customer.c_name FROM (SELECT customer.c_custkey, customer.c_name FROM customer AS customer) AS customer) AS customer_view"
    );
    Ok(())
}

#[test]
fn test_unparse_left_anti_join() -> Result<()> {
    // select t1.d from t1 where c not in (select c from t2)
    let schema = Schema::new(vec![
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ]);

    // LeftAnti Join: t1.c = __correlated_sq_1.c
    //   TableScan: t1 projection=[c]
    //   SubqueryAlias: __correlated_sq_1
    //     TableScan: t2 projection=[c]

    let table_scan1 = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let table_scan2 = table_scan(Some("t2"), &schema, Some(vec![0]))?.build()?;
    let subquery = subquery_alias(table_scan2, "__correlated_sq_1")?;
    let plan = LogicalPlanBuilder::from(table_scan1)
        .project(vec![col("t1.d")])?
        .join_on(
            subquery,
            datafusion_expr::JoinType::LeftAnti,
            vec![col("t1.c").eq(col("__correlated_sq_1.c"))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t1"."d" FROM "t1" WHERE NOT EXISTS (SELECT 1 FROM "t2" AS "__correlated_sq_1" WHERE ("t1"."c" = "__correlated_sq_1"."c"))"#
    );
    Ok(())
}

#[test]
fn test_unparse_left_semi_join() -> Result<()> {
    // select t1.d from t1 where c in (select c from t2)
    let schema = Schema::new(vec![
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ]);

    // LeftSemi Join: t1.c = __correlated_sq_1.c
    //   TableScan: t1 projection=[c]
    //   SubqueryAlias: __correlated_sq_1
    //     TableScan: t2 projection=[c]

    let table_scan1 = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let table_scan2 = table_scan(Some("t2"), &schema, Some(vec![0]))?.build()?;
    let subquery = subquery_alias(table_scan2, "__correlated_sq_1")?;
    let plan = LogicalPlanBuilder::from(table_scan1)
        .project(vec![col("t1.d")])?
        .join_on(
            subquery,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.c").eq(col("__correlated_sq_1.c"))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t1"."d" FROM "t1" WHERE EXISTS (SELECT 1 FROM "t2" AS "__correlated_sq_1" WHERE ("t1"."c" = "__correlated_sq_1"."c"))"#
    );
    Ok(())
}

/// The schema the bounded-`EXISTS` cases below join on.
fn exists_fetch_schema() -> Schema {
    Schema::new(vec![
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ])
}

/// A flat schema of non-nullable `Int32` columns, for a build side whose column
/// *names* are what a test varies.
fn int32_schema(names: &[&str]) -> Schema {
    Schema::new(
        names
            .iter()
            .map(|name| Field::new(*name, DataType::Int32, false))
            .collect::<Vec<_>>(),
    )
}

/// The probe side shared by the unqualified-correlation tests: `p`, projected so
/// that both of its output columns carry a bare name.
///
/// That projection is what leaves the join keys built from it unqualified, which
/// is the whole point — a correlation with no qualifier collides on its column
/// name instead.
fn unqualified_probe() -> Result<LogicalPlan> {
    table_scan(Some("p"), &exists_fetch_schema(), Some(vec![0, 1]))?
        .project(vec![col("p.c").alias("c"), col("p.d").alias("d")])?
        .build()
}

/// Builds `<join_type> Join: t1.c = t2.c` with `fetch` rows read from the
/// build side, projecting `t1.d`.
fn exists_join_with_build_side_fetch(
    join_type: datafusion_expr::JoinType,
    fetch: Option<usize>,
) -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan_with_filter_and_fetch(
        Some("t2"),
        &schema,
        Some(vec![0]),
        vec![],
        fetch,
    )?
    .build()?;

    LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d")])?
        .join_on(build, join_type, vec![col("t1.c").eq(col("t2.c"))])?
        .build()
}

/// A `fetch` on a semi join's build side bounds how much of that side the join
/// reads, so the correlation has to be applied to the bounded rows. Emitting it
/// beside the correlation in the `EXISTS` body instead makes `LIMIT` a no-op —
/// the body is non-empty whenever *any* `t2` row matches, not only when one of
/// the read rows does — and the semi join reports matches the plan never read.
#[test]
fn test_unparse_left_semi_join_scopes_build_side_fetch() -> Result<()> {
    let plan =
        exists_join_with_build_side_fetch(datafusion_expr::JoinType::LeftSemi, Some(5))?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d" FROM "t1" WHERE EXISTS (SELECT 1 FROM (SELECT "t2"."c" FROM "t2" LIMIT 5) AS "t2" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

/// The anti join is the mirror image: the unscoped `LIMIT` makes the body
/// non-empty for matches outside the read rows, so `NOT EXISTS` drops rows it
/// should have returned.
#[test]
fn test_unparse_left_anti_join_scopes_build_side_fetch() -> Result<()> {
    let plan =
        exists_join_with_build_side_fetch(datafusion_expr::JoinType::LeftAnti, Some(5))?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d" FROM "t1" WHERE NOT EXISTS (SELECT 1 FROM (SELECT "t2"."c" FROM "t2" LIMIT 5) AS "t2" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

/// A mark join carries the same `EXISTS` body into the boolean it projects, so
/// the bound needs the same scope there — an unscoped `LIMIT` marks a row true
/// on the strength of a `t2` row the plan never read.
#[test]
fn test_unparse_left_mark_join_scopes_build_side_fetch() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan_with_filter_and_fetch(
        Some("t2"),
        &schema,
        Some(vec![0]),
        vec![],
        Some(5),
    )?
    .build()?;

    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftMark,
            vec![col("t1.c").eq(col("t2.c"))],
        )?
        .project(vec![col("t1.d")])?
        .filter(col("mark").or(col("t1.d").lt(lit(0))))?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d" FROM "t1" WHERE (EXISTS (SELECT 1 FROM (SELECT "t2"."c" FROM "t2" LIMIT 5) AS "t2" WHERE ("t1"."c" = "t2"."c")) OR ("t1"."d" < 0))"#
    );
    Ok(())
}

/// A right semi join swaps its inputs before the `EXISTS` body is built, so the
/// side being scoped is `join.left` and the correlation names `join.right`. The
/// scope's name has to follow that swap: taking it from the unswapped side
/// aliases the bounded `t1` rows as `t2` and leaves the correlation naming a
/// `t1` that is no longer in scope, so the subquery does not bind at all.
#[test]
fn test_unparse_right_semi_join_scopes_build_side_fetch() -> Result<()> {
    let plan = exists_join_with_probe_side_fetch(datafusion_expr::JoinType::RightSemi)?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t2"."d" FROM "t2" WHERE EXISTS (SELECT 1 FROM (SELECT "t1"."c" FROM "t1" LIMIT 5) AS "t1" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

/// The right anti join takes the same swap, and inherits it from the same place.
#[test]
fn test_unparse_right_anti_join_scopes_build_side_fetch() -> Result<()> {
    let plan = exists_join_with_probe_side_fetch(datafusion_expr::JoinType::RightAnti)?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t2"."d" FROM "t2" WHERE NOT EXISTS (SELECT 1 FROM (SELECT "t1"."c" FROM "t1" LIMIT 5) AS "t1" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

/// Builds `<join_type> Join: t1.c = t2.c` with the bound on `t1`, the input a
/// right semi or anti join correlates *out of* — its build side.
fn exists_join_with_probe_side_fetch(
    join_type: datafusion_expr::JoinType,
) -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let bounded = table_scan_with_filter_and_fetch(
        Some("t1"),
        &schema,
        Some(vec![0]),
        vec![],
        Some(5),
    )?
    .build()?;
    let correlated = table_scan(Some("t2"), &schema, Some(vec![0, 1]))?.build()?;

    LogicalPlanBuilder::from(bounded)
        .join_on(correlated, join_type, vec![col("t1.c").eq(col("t2.c"))])?
        .project(vec![col("t2.d")])?
        .build()
}

/// Builds `LeftSemi Join: t1.c = s.t2.c` with the build side `s.t2` qualified,
/// optionally bounded, for the fully-qualified-dialect cases below.
fn qualified_build_side_semi_join(fetch: Option<usize>) -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan_with_filter_and_fetch(
        Some(TableReference::partial("s", "t2")),
        &schema,
        Some(vec![0]),
        vec![],
        fetch,
    )?
    .build()?;

    LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.c").eq(col("s.t2.c"))],
        )?
        .build()
}

/// A dialect that spells columns in full writes every component of a qualified
/// name in the correlated predicate, but a derived table's alias is a single
/// identifier and can only carry the last one, so the bound cannot be scoped.
///
/// Every shape `exists_scope_name` cannot name is refused, this one included:
/// leaving the bound unscoped binds and runs, silently answering from rows
/// outside the bound. Losing the pushdown is the cheaper failure, and it is the
/// trade `derive_row_limited_scope` already makes.
#[test]
fn test_unparse_semi_join_build_side_fetch_refuses_qualified_full_column_dialect()
-> Result<()> {
    let plan = qualified_build_side_semi_join(Some(5))?;

    let dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .build();
    let err = Unparser::new(&dialect)
        .plan_to_sql(&plan)
        .expect_err("a bound that cannot be scoped must not unparse");
    assert_contains!(
        err.to_string(),
        "not supported for a qualified table name on a dialect that spells columns in full"
    );
    Ok(())
}

/// The refusal above is owed only to the bound. Without one there is nothing to
/// scope and nothing to get wrong, so the same plan and dialect must still
/// unparse — which is why the bound is tested for before the scope is named.
#[test]
fn test_unparse_semi_join_without_fetch_keeps_qualified_full_column_dialect() -> Result<()>
{
    let plan = qualified_build_side_semi_join(None)?;

    let dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .build();
    assert_snapshot!(
        Unparser::new(&dialect).plan_to_sql(&plan)?,
        @r#"SELECT t1.d FROM t1 WHERE EXISTS (SELECT 1 FROM s.t2 WHERE (t1.c = s.t2.c))"#
    );
    Ok(())
}

/// A `Limit` node above the build side bounds it just as a scan `fetch` does,
/// and `OFFSET` picks *which* rows as much as `LIMIT` does — both belong inside
/// the scope, with the correlation outside it.
#[test]
fn test_unparse_left_semi_join_scopes_build_side_limit_node() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t2"), &schema, Some(vec![0]))?
        .limit(2, Some(5))?
        .build()?;

    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.c").eq(col("t2.c"))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d" FROM "t1" WHERE EXISTS (SELECT 1 FROM (SELECT "t2"."c" FROM "t2" LIMIT 5 OFFSET 2) AS "t2" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

/// A build side that is a set operation is already wrapped as a derived table
/// to serve as the `EXISTS` body. Bounding it adds a second scope around that
/// one, and the inner select then has to project the union's columns rather
/// than the `SELECT 1` the outermost body wants.
#[test]
fn test_unparse_left_semi_join_scopes_bounded_set_operation_build_side() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t2"), &schema, Some(vec![0]))?
        .union(table_scan(Some("t3"), &schema, Some(vec![0]))?.build()?)?
        .limit(0, Some(5))?
        .build()?;

    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.c").eq(col("t2.c"))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?.to_string();
    assert!(
        !sql.contains("SELECT  FROM") && !sql.contains("SELECT FROM"),
        "the scoped body must project something: {sql}"
    );
    assert_snapshot!(
        sql,
        @r#"SELECT "t1"."d" FROM "t1" WHERE EXISTS (SELECT 1 FROM (SELECT * FROM (SELECT "t2"."c" FROM "t2" UNION ALL SELECT "t3"."c" FROM "t3") LIMIT 5) AS "t2" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

/// Builds `LeftSemi Join: t1.c = t2.c AND t1.d = t3.d` over a `t2 INNER JOIN t3`
/// build side, optionally bounded, so the bounded and unbounded cases below
/// differ only in the bound.
fn multi_relation_build_side_semi_join(fetch: Option<usize>) -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t2"), &schema, Some(vec![0]))?.join_on(
        table_scan(Some("t3"), &schema, Some(vec![0, 1]))?.build()?,
        datafusion_expr::JoinType::Inner,
        vec![col("t2.c").eq(col("t3.c"))],
    )?;
    // Applied only when asked: `limit(0, None)` still inserts a `Limit` node,
    // and the unbounded cases are about a build side that carries no bound.
    let build = match fetch {
        Some(fetch) => build.limit(0, Some(fetch))?,
        None => build,
    }
    .build()?;

    LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.c").eq(col("t2.c")), col("t1.d").eq(col("t3.d"))],
        )?
        .build()
}

/// Builds `<join_type> Join: t.c = t.c` where the build side is the same relation
/// as the probe, optionally bounded — the self-join whose correlation carries
/// only a qualifier the probe also answers to.
fn probe_qualified_self_join(
    fetch: Option<usize>,
    join_type: datafusion_expr::JoinType,
) -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan_with_filter_and_fetch(
        Some("t"),
        &schema,
        Some(vec![0]),
        vec![],
        fetch,
    )?
    .build()?;

    LogicalPlanBuilder::from(probe)
        .project(vec![col("t.d")])?
        .join_on(build, join_type, vec![col("t.c").eq(col("t.c"))])?
        .build()
}

/// Builds `LeftSemi Join: t1.c = t3.c` where the probe is `t1 INNER JOIN t` and
/// the build side is `t INNER JOIN t3`, so the two sides share the qualifier `t`
/// while the correlation names neither of them.
///
/// The shared qualifier is what makes capture *possible*; this plan separates
/// that from a correlation actually being captured.
fn overlapping_but_unreferenced_qualifier_semi_join() -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?
        .join_on(
            table_scan(Some("t"), &schema, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            vec![col("t1.c").eq(col("t.c"))],
        )?
        .build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![0]))?
        .join_on(
            table_scan(Some("t3"), &schema, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            vec![col("t.c").eq(col("t3.c"))],
        )?
        .build()?;

    // `t.c` is kept in the projection deliberately: projecting every column of
    // `t` away would leave the probe's schema carrying no `t` qualifier at all,
    // and the overlap this plan exists to exercise would not be there.
    LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d"), col("t.c")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.c").eq(col("t3.c"))],
        )?
        .build()
}

/// A correlation naming more than one of the build side's own inputs cannot be
/// scoped: one derived table can answer to only one of those names, so no name
/// keeps every reference bound to the relation it came from.
///
/// Leaving the bound beside the correlation is not the safe fallback it looks
/// like. That output is valid SQL — every qualifier still binds, `t1` to the
/// outer query and `t2`/`t3` to the subquery's own inputs — so the database runs
/// it and answers from rows outside the bound, which is the #12595 defect this
/// shape was left holding. Refusing costs the pushdown instead of returning
/// wrong rows.
///
/// Repairing it needs the correlation's qualifiers rewritten to the new scope,
/// tracked by spiceai/spiceai#12840; whoever implements that should expect this
/// refusal to become a scope.
#[test]
fn test_unparse_left_semi_join_refuses_multi_relation_build_side() -> Result<()> {
    let plan = multi_relation_build_side_semi_join(Some(5))?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let err = unparser
        .plan_to_sql(&plan)
        .expect_err("a correlation naming two build-side inputs must be refused");
    assert_snapshot!(
        err,
        @"This feature is not implemented: Unparsing a row bound on an EXISTS-style join's build side is not supported when the correlation names more than one of the build side's inputs"
    );
    Ok(())
}

/// A correlation whose only qualifier is one the probe side also answers to is
/// captured by the subquery's own `FROM "t"`: `"t"."c" = "t"."c"` binds entirely
/// to the inner relation, so the correlation is lost, the `EXISTS` reduces to
/// "this table has a row", and the semi join keeps every probe row. That is
/// valid SQL, so a database runs it and returns those wrong rows.
///
/// Scoping the bound under an invented name does not repair it either — that
/// would rebind the references to the probe, trading one wrong answer for
/// another — so the shape is refused.
///
/// The bounded case is refused by the same bound-independent check as
/// [`test_unparse_left_semi_join_without_fetch_refuses_shadowed_correlation`],
/// which is why both report capture rather than a bound that cannot be scoped.
#[test]
fn test_unparse_left_semi_join_refuses_probe_qualified_correlation() -> Result<()> {
    let plan = probe_qualified_self_join(Some(5), datafusion_expr::JoinType::LeftSemi)?;

    assert_captured_correlation_refused(
        &plan,
        "a probe-qualified correlation must be refused, not shadowed",
    );
    Ok(())
}

/// The same multi-relation correlation, unbounded: no scope is demanded, so the
/// correlation stays in the body's own `WHERE` where it is correct, and both
/// `t2` and `t3` are still in scope there.
///
/// Pins that the refusal above is gated on the bound rather than on the shape of
/// the correlation. Widening it to every multi-relation correlation would cost
/// the pushdown on queries that unparse correctly today.
#[test]
fn test_unparse_left_semi_join_without_fetch_keeps_multi_relation_correlation()
-> Result<()> {
    let plan = multi_relation_build_side_semi_join(None)?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d" FROM "t1" WHERE EXISTS (SELECT 1 FROM "t2" INNER JOIN "t3" ON ("t2"."c" = "t3"."c") WHERE (("t1"."c" = "t2"."c") AND ("t1"."d" = "t3"."d")))"#
    );
    Ok(())
}

/// The refusal every captured-correlation shape must produce, spelled once.
///
/// Asserted rather than snapshotted: an inline snapshot is keyed by the location
/// of its macro, so one shared by several tests collides between them. Every
/// caller has to produce this identical string, which is what makes sharing it
/// the point — a reword stays a one-place edit.
const CAPTURED_CORRELATION_REFUSAL: &str = "This feature is not implemented: Unparsing an EXISTS-style join is not supported when a FROM the emitted SQL introduces would capture the correlation: it answers to the correlated reference's relation qualifier, or exposes its column name when the reference carries none, or is a relation this unparser cannot read at all, so the reference binds there instead of in the query it was written against";

#[track_caller]
fn assert_captured_correlation_refused(plan: &LogicalPlan, context: &str) {
    assert_captured_correlation_refused_by(
        &Unparser::new(&UnparserPostgreSqlDialect {}),
        plan,
        context,
    );
}

/// [`assert_captured_correlation_refused`] for a shape whose dialect is the point.
#[track_caller]
fn assert_captured_correlation_refused_by(
    unparser: &Unparser,
    plan: &LogicalPlan,
    context: &str,
) {
    let err = unparser.plan_to_sql(plan).expect_err(context);
    assert_eq!(err.to_string(), CAPTURED_CORRELATION_REFUSAL);
}

/// The capture does not need a bound to bite. With no bound at all the same
/// plan used to unparse to
/// `... WHERE EXISTS (SELECT 1 FROM "t" WHERE ("t"."c" = "t"."c"))`, whose inner
/// `FROM "t"` shadows the outer `"t"` — valid SQL that runs and answers from an
/// inner tautology, so a semi or mark join keeps every probe row and an anti
/// join drops every one.
///
/// Nothing about that depends on a row bound, so the refusal is decided by the
/// correlation's qualifiers rather than by the bound, and the unbounded shape is
/// refused on the same terms as the bounded one above.
///
/// Repairing it rather than refusing it needs the correlated qualifiers
/// rewritten to the scope the derived table introduces, tracked by
/// spiceai/spiceai#12840.
#[test]
fn test_unparse_left_semi_join_without_fetch_refuses_shadowed_correlation() -> Result<()>
{
    let plan = probe_qualified_self_join(None, datafusion_expr::JoinType::LeftSemi)?;

    assert_captured_correlation_refused(
        &plan,
        "an unbounded shadowed correlation must be refused, not emitted",
    );
    Ok(())
}

/// The anti join's harm runs the other way and is covered by the same refusal.
///
/// Captured, the correlation is an inner tautology, so `NOT EXISTS` is false for
/// every probe row and the join returns nothing — where a semi join keeps rows it
/// should not, an anti join drops rows it should return. Both are wrong rows, so
/// neither may be emitted.
#[test]
fn test_unparse_left_anti_join_without_fetch_refuses_shadowed_correlation() -> Result<()>
{
    let plan = probe_qualified_self_join(None, datafusion_expr::JoinType::LeftAnti)?;

    assert_captured_correlation_refused(
        &plan,
        "an unbounded shadowed anti-join correlation must be refused",
    );
    Ok(())
}

/// The right semi join reaches the same refusal, covering the join types that
/// swap which input is correlated against.
///
/// The check does not consult that swap — a shared qualifier is shared whichever
/// side is read first — so this covers the join type rather than pinning an
/// orientation. It is here because the swap is easy to reintroduce while
/// reworking this code, and reaching the refusal from both orientations is what
/// says it is not needed.
#[test]
fn test_unparse_right_semi_join_without_fetch_refuses_shadowed_correlation() -> Result<()>
{
    let plan = probe_qualified_self_join(None, datafusion_expr::JoinType::RightSemi)?;

    assert_captured_correlation_refused(
        &plan,
        "a swapped shadowed correlation must be refused too",
    );
    Ok(())
}

/// Builds `LeftSemi Join` correlating `<probe>.c > 0` against a build side aliased
/// `s.a`, which the unparser emits as the single identifier `AS "a"`.
fn qualified_alias_build_side_semi_join(probe_name: &str) -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some(probe_name), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t2"), &schema, Some(vec![0]))?
        .alias("s.a")?
        .build()?;

    LogicalPlanBuilder::from(probe)
        .project(vec![col(format!("{probe_name}.d"))])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col(format!("{probe_name}.c")).gt(lit(0))],
        )?
        .build()
}

/// A derived table's alias is a single identifier however the dialect spells
/// columns, so a *qualified* alias captures under only its last component.
///
/// The build side is aliased `s.a` and emitted `AS "a"`, so an outer relation
/// named `a` is shadowed. Keying the alias off the output schema — which carries
/// the whole `s.a` reference — misses this, because on a fully-qualifying dialect
/// it looks for `s.a` while the SQL only ever says `a`.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_qualified_alias_last_component()
-> Result<()> {
    let plan = qualified_alias_build_side_semi_join("a")?;

    let dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .build();
    let err = Unparser::new(&dialect)
        .plan_to_sql(&plan)
        .expect_err("an alias emitted as `AS a` must be seen to shadow the outer `a`");
    assert_eq!(err.to_string(), CAPTURED_CORRELATION_REFUSAL);
    Ok(())
}

/// The other direction: `AS "a"` does not shadow an outer `s.a`, so that keeps its
/// pushdown.
///
/// `s.a.c` names schema `s`, table `a`; the alias is an unqualified `a` and does
/// not answer to it, so the reference binds outward. Keying the alias off the
/// schema would have matched `s.a` here and refused correct SQL — the same
/// mismatch as the test above, costing pushdown instead of correctness.
#[test]
fn test_unparse_left_semi_join_keeps_qualified_reference_past_bare_alias() -> Result<()> {
    let plan = qualified_alias_build_side_semi_join("s.a")?;

    let dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .build();
    assert_snapshot!(
        Unparser::new(&dialect).plan_to_sql(&plan)?,
        @"SELECT s.a.d FROM s.a WHERE EXISTS (SELECT 1 FROM t2 AS a WHERE (s.a.c > 0))"
    );
    Ok(())
}

/// An alias the unparser invents for a derived table is in the emitted scope even
/// though it appears nowhere in the plan.
///
/// A relation a user named `derived_limit` cannot be distinguished from the alias
/// `exists_scope_name` hands a bounded build side, so a correlation naming it is
/// refused rather than risk the invented alias capturing it. Refusing costs the
/// pushdown on a pathological table name; emitting risks wrong rows.
#[test]
fn test_unparse_left_semi_join_refuses_correlation_naming_an_invented_alias() -> Result<()>
{
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("derived_limit"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t2"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("derived_limit.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("derived_limit.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a correlation naming an alias the unparser can invent must be refused",
    );
    Ok(())
}

/// Builds `LeftSemi Join` correlating `<probe_name>.c > 0` against an unrelated
/// build side, so the only thing that can capture the reference is an alias the
/// unparser invents for a derived table of its own.
fn invented_alias_collision_semi_join(probe_name: &str) -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some(probe_name), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t2"), &schema, Some(vec![0]))?.build()?;

    LogicalPlanBuilder::from(probe)
        .project(vec![col(format!("{probe_name}.d"))])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col(format!("{probe_name}.c")).gt(lit(0))],
        )?
        .build()
}

/// The aliases the unparser numbers carry the number in the emitted SQL, so that
/// is the name a correlation collides with.
///
/// `SelectBuilder::next_derived_aggregate_alias` emits `derived_aggregate_1`,
/// `derived_aggregate_2`, … — never a bare `derived_aggregate` — so a build side
/// stacking an aggregate writes `AS derived_aggregate_1`, and a relation a user
/// named that is captured by it.
#[test]
fn test_unparse_left_semi_join_refuses_correlation_naming_a_numbered_alias() -> Result<()>
{
    let plan = invented_alias_collision_semi_join("derived_aggregate_1")?;

    assert_captured_correlation_refused(
        &plan,
        "a correlation naming a numbered alias the unparser can invent must be refused",
    );
    Ok(())
}

/// The LATERAL FLATTEN aliases are numbered the same way and reach the emitted
/// `FROM` the same way, so they are reserved on the same terms.
#[test]
fn test_unparse_left_semi_join_refuses_correlation_naming_a_numbered_flatten_alias()
-> Result<()> {
    let plan = invented_alias_collision_semi_join("_unnest_2")?;

    assert_captured_correlation_refused(
        &plan,
        "a correlation naming a numbered FLATTEN alias must be refused",
    );
    Ok(())
}

/// The other direction: the bare prefix is not a name the unparser ever emits,
/// so a relation called `derived_aggregate` keeps its pushdown.
///
/// Reserving the prefix rather than the generated form gets this exactly
/// backwards — it costs the pushdown here, where nothing can capture, while
/// leaving `derived_aggregate_1` open, where something can.
#[test]
fn test_unparse_left_semi_join_keeps_correlation_naming_an_unnumbered_prefix()
-> Result<()> {
    let plan = invented_alias_collision_semi_join("derived_aggregate")?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "derived_aggregate"."d" FROM "derived_aggregate" WHERE EXISTS (SELECT 1 FROM "t2" WHERE ("derived_aggregate"."c" > 0))"#
    );
    Ok(())
}

/// An outer reference is emitted as the same qualified identifier as a plain
/// column, so it is captured the same way and has to be refused the same way.
///
/// `Expr::OuterReferenceColumn` renders through `col_to_sql`, exactly as
/// `Expr::Column` does, so `"t"."c" > 0` lands inside the `EXISTS` body either
/// way and binds to the build side's own `t`. `Expr::column_refs` collects only
/// the plain variant, so a guard built on it walks past this one.
///
/// The predicate is put in the join's filter because that is where
/// `LogicalPlanBuilder` puts a predicate it cannot attribute to a side, and an
/// outer reference names no column either side owns.
#[test]
fn test_unparse_left_semi_join_refuses_shadowed_outer_reference() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?
        .join_on(
            table_scan(Some("t"), &schema, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            vec![col("t1.c").eq(col("t.c"))],
        )?
        .build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d"), col("t.c")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![out_ref_col(DataType::Int32, "t.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a shadowed outer reference must be refused, not emitted",
    );
    Ok(())
}

/// Builds `LeftSemi Join` on an unqualified key: the probe projects `p.c AS c`,
/// and the build side is a relation whose own columns are `build_columns`,
/// joined on its first one.
///
/// The correlation therefore carries no qualifier, and whether the emitted body
/// captures it turns entirely on whether the build relation has a column called
/// `c` — which `build_columns` decides.
fn unqualified_correlation_semi_join(build_columns: &[&str]) -> Result<LogicalPlan> {
    let probe = unqualified_probe()?;
    let build_schema = int32_schema(build_columns);
    let build = table_scan(Some("b"), &build_schema, Some(vec![0]))?.build()?;

    LogicalPlanBuilder::from(probe)
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["c"], vec![build_columns[0]]),
            None,
        )?
        .build()
}

/// An unqualified correlated reference is captured by whichever relation in the
/// body exposes its column name, and needs no shared relation to be captured.
///
/// The two sides here are `p` and `b`, which share no relation name — but the
/// probe projects its column to a bare `c` and `b` has a column called `c`, so
/// the body's own `FROM "b"` answers to the reference. Before this was caught,
/// the plan unparsed to
/// `SELECT "p"."c" AS "c", "p"."d" AS "d" FROM "p" WHERE EXISTS (SELECT 1 FROM "b" WHERE ("c" = "c"))`,
/// where both halves bind to `b.c`: the same inner tautology as the self-join,
/// reached without either side naming the other.
///
/// So the scope has to carry the column names the emitted `FROM` exposes and
/// not only the relations it introduces — a qualifier is simply not what an
/// unqualified reference collides with.
#[test]
fn test_unparse_left_semi_join_refuses_unqualified_correlation() -> Result<()> {
    let plan = unqualified_correlation_semi_join(&["c", "d"])?;

    assert_captured_correlation_refused(
        &plan,
        "an unqualified correlation the build side exposes must be refused",
    );
    Ok(())
}

/// The other direction: an unqualified reference to a name the body does not
/// expose binds outward, so it keeps its pushdown.
///
/// Identical to the test above except that `b`'s columns are `e` and `f`. The
/// correlated `c` matches nothing the subquery's `FROM` answers to, so it
/// resolves against the outer query and the SQL is emitted.
///
/// The name has to be absent from the *relation*, not merely unprojected: `b`
/// is emitted bare, so `FROM "b"` exposes every column it has whatever the plan
/// projects — which is why the build side here is a different relation rather
/// than the same one projected differently.
#[test]
fn test_unparse_left_semi_join_keeps_unqualified_reference_the_body_lacks() -> Result<()>
{
    let plan = unqualified_correlation_semi_join(&["e", "f"])?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "p"."c" AS "c", "p"."d" AS "d" FROM "p" WHERE EXISTS (SELECT 1 FROM "b" WHERE ("c" = "b"."e"))"#
    );
    Ok(())
}

/// An unqualified reference in the join's filter that only the build side can
/// mean is a build-side reference, binds inside on purpose, and keeps its
/// pushdown.
///
/// `join.filter` carries no split by side, but it can only reference the join's
/// own two inputs — so a name just one of them exposes is attributable after
/// all. Here `e` belongs to `b` alone, and `("e" > 0)` inside the body is meant
/// for the body's own relation. Testing filter names against the build side
/// outright, rather than against the names both sides answer to, refuses this.
///
/// The qualifier arm is restricted the same way and for the same reason; this
/// is that restriction for names.
#[test]
fn test_unparse_left_semi_join_keeps_build_only_unqualified_filter_name() -> Result<()> {
    let probe = unqualified_probe()?;
    let build_schema = int32_schema(&["e", "f"]);
    // Aliased so the build side's output field is unqualified, which is what
    // leaves the filter's reference to it unqualified too.
    let build = table_scan(Some("b"), &build_schema, Some(vec![0]))?
        .project(vec![col("b.e").alias("e")])?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("e").gt(lit(0))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "p"."c" AS "c", "p"."d" AS "d" FROM "p" WHERE EXISTS (SELECT 1 FROM "b" WHERE ("e" > 0))"#
    );
    Ok(())
}

/// A build relation exposes every column it has, not the projected ones, so an
/// unqualified correlation collides with a name the plan never selects.
///
/// `b` is scanned for `d` alone, but it is emitted bare as `FROM "b"`, and a
/// bare relation answers to all of its columns. Reading the exposed names off
/// the build side's *output* schema reports only `d`, so the correlated `c`
/// reads as unclaimed; the SQL says otherwise. Un-guarded this emits
/// `... EXISTS (SELECT 1 FROM "b" WHERE ("c" = "b"."d"))`, where `"c"` binds to
/// `b.c` and the correlation is gone.
///
/// This is the unqualified counterpart of
/// [`test_unparse_left_semi_join_refuses_capture_by_unprojected_build_relation`],
/// and it is why the scan's own schema is collected as well as the plan's.
#[test]
fn test_unparse_left_semi_join_refuses_unqualified_capture_by_unprojected_column()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = unqualified_probe()?;
    let build = table_scan(Some("b"), &schema, Some(vec![1]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["c"], vec!["b.d"]),
            None,
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a build column the plan does not project must still be seen to capture",
    );
    Ok(())
}

/// The mirror case: a derived table the unparser invents exposes a name that
/// belongs to no relation in the plan at all.
///
/// `b`'s columns are `x` and `y`; the build side renames `x` to `c` and stacks
/// enough operators that the emitter wraps it, so the body reads
/// `FROM (SELECT "b"."x" AS "c" FROM "b") AS "derived_projection"` — and that
/// derived table answers to `c`. Un-guarded the whole predicate is
/// `("c" = "c")`, both halves binding to `derived_projection.c`: the inner
/// tautology again, so a semi join keeps every probe row.
///
/// No walk over the plan's relations can find that name — it is not a column of
/// `b` — which is why the exposed names start from the build plan's own output
/// schema. A derived table exposes exactly that, whatever it was renamed from.
#[test]
fn test_unparse_left_semi_join_refuses_unqualified_capture_by_renamed_column()
-> Result<()> {
    let probe = unqualified_probe()?;
    let build_schema = int32_schema(&["x", "y"]);
    let build = table_scan(Some("b"), &build_schema, Some(vec![0]))?
        .project(vec![col("b.x").alias("c")])?
        .project(vec![col("c")])?
        .distinct()?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["c"], vec!["c"]),
            None,
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a name a derived table renames into the body must be seen to capture",
    );
    Ok(())
}
/// A quoted single-part table relation, the shape every mock below writes.
fn quoted_table(name: &str) -> TableRelationBuilder {
    let mut builder = TableRelationBuilder::default();
    builder.name(ObjectName::from(vec![Ident::with_quote('"', name)]));
    builder
}

/// An extension unparser that writes a table relation of its own choosing into
/// the emitted `FROM`, and optionally cross-joins a second one.
///
/// Both are ordinary uses of what [`UserDefinedLogicalNodeUnparser`] hands out —
/// the `RelationBuilder` for the first, the `SelectBuilder` for the second — and
/// together they are the whole latitude that makes an extension's emitted scope
/// unreadable from the plan: `emitted_scope` walks the inputs, which say nothing
/// about either name.
///
/// `joined` is the axis the tests below differ on, which is why it is a field
/// rather than a second implementation: an enclosing `SubqueryAlias` aliases
/// `relation` alone, so only the joined relation keeps its own name under one.
struct MockRelationWritingUnparser {
    relation: &'static str,
    joined: Option<&'static str>,
}

impl UserDefinedLogicalNodeUnparser for MockRelationWritingUnparser {
    fn unparse(
        &self,
        node: &dyn UserDefinedLogicalNode,
        _unparser: &Unparser,
        _query: &mut Option<&mut QueryBuilder>,
        select: &mut Option<&mut SelectBuilder>,
        relation: &mut Option<&mut RelationBuilder>,
    ) -> Result<UnparseWithinStatementResult> {
        if node
            .as_any()
            .downcast_ref::<MockUserDefinedLogicalPlan>()
            .is_none()
        {
            return Ok(UnparseWithinStatementResult::Unmodified);
        }
        if let Some(rel) = relation {
            rel.table(quoted_table(self.relation));
        }

        // A join rather than a second `push_from`: the emitter treats the FROM
        // list as a stack it pops from and pushes back, so an extra entry makes
        // it pop the wrong one and fail loudly instead. A join is the shape that
        // emits.
        if let Some(joined) = self.joined {
            let cross = sqlparser::ast::Join {
                relation: quoted_table(joined)
                    .build()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
                global: false,
                join_operator: sqlparser::ast::JoinOperator::CrossJoin(
                    sqlparser::ast::JoinConstraint::None,
                ),
            };
            if let Some(sel) = select
                && let Some(mut from) = sel.pop_from()
            {
                from.push_join(cross);
                sel.push_from(from);
            }
        }
        Ok(UnparseWithinStatementResult::Modified)
    }
}

/// An extension node whose inputs expose `columns` unqualified, so a join key
/// built from one of them is an unqualified reference into the emitted body.
fn unqualified_extension(columns: &[&str]) -> Result<LogicalPlan> {
    let input = LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: false,
        schema: Arc::new(DFSchema::try_from(int32_schema(columns))?),
    });
    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(MockUserDefinedLogicalPlan { input }),
    }))
}

/// An [`Unparser`] whose extension nodes emit `relation`, plus `joined` if given.
fn extension_unparser(
    relation: &'static str,
    joined: Option<&'static str>,
) -> Unparser<'static> {
    Unparser::new(&UnparserPostgreSqlDialect {}).with_extension_unparsers(vec![Arc::new(
        MockRelationWritingUnparser { relation, joined },
    )])
}

/// `LeftSemi Join` correlating the probe `t` against `build` on `t.c`, with the
/// build half of the pair keyed by `build_key`.
fn extension_capture_semi_join(
    build: LogicalPlan,
    build_key: &str,
) -> Result<LogicalPlan> {
    let probe = table_scan(Some("t"), &exists_fetch_schema(), Some(vec![0, 1]))?
        .project(vec![col("t.d")])?
        .build()?;
    LogicalPlanBuilder::from(probe)
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["t.c"], vec![build_key]),
            None,
        )?
        .build()
}

/// An extension node decides its own emitted `FROM`, so the relation it
/// introduces cannot be read from the plan — and the correlation has to be
/// refused rather than cleared against a scope nobody looked at.
///
/// The probe is `t`; the build side is an extension whose *inputs* name no
/// relation at all and whose unparser writes `FROM "t"`. Every part of the walk
/// is misled at once: it finds no qualifier to compare `t.c` with, and the one
/// name it would have needed is not in the plan to be found. Before this was
/// caught, the plan unparsed to
/// `SELECT "t"."d" FROM "t" WHERE EXISTS (SELECT 1 FROM "t" WHERE ("t"."c" = "c"))`,
/// where the outer `"t"."c"` binds to the extension's own `t` and the build half
/// `"c"` binds to the same row: the inner tautology again, so the semi join
/// keeps every probe row.
///
/// Refusing costs the pushdown for every extension on a correlated side,
/// including the ones that would have been correct. Narrowing it needs the
/// extension unparser to report the scope it will emit, which the trait has no
/// way to say.
#[test]
fn test_unparse_left_semi_join_refuses_correlation_captured_by_extension_relation()
-> Result<()> {
    let plan = extension_capture_semi_join(unqualified_extension(&["c"])?, "c")?;

    assert_captured_correlation_refused_by(
        &extension_unparser("t", None),
        &plan,
        "a relation an extension emits must be refused, not cleared unseen",
    );
    Ok(())
}

/// A `SubqueryAlias` over an extension does not make the extension readable, so
/// the alias cannot be taken as the whole of what the body's `FROM` answers to.
///
/// The walk stops at a `SubqueryAlias` on purpose — an alias replaces the name
/// it is given to, so the relations below it are not addressable through it.
/// That reasoning holds for relations the walk can see, and an extension is
/// exactly the one it cannot: the alias reaches the single relation the
/// `RelationBuilder` holds, while the same unparser is handed the
/// `SelectBuilder` and can join a second relation onto the same `FROM`.
///
/// Here the extension writes `safe` for the alias to take and cross-joins `t`,
/// which is also the probe. Before this was caught, the plan unparsed to
/// `SELECT "t"."d" FROM "t" WHERE EXISTS (SELECT 1 FROM "safe" AS "a" CROSS JOIN "t" WHERE ("t"."c" = "a"."k"))`,
/// where `"t"."c"` binds to the joined-in `t` rather than the outer one: the
/// correlation is gone, the `EXISTS` is uncorrelated, and the semi join keeps
/// every probe row whenever that cross join has a row at all.
///
/// This costs the pushdown for an aliased extension that only ever writes its
/// own relation, which would have been correct — the guard runs before any of
/// the body is built, so there is no emitted `FROM` to tell the two apart.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_an_aliased_extensions_join()
-> Result<()> {
    let build = subquery_alias(unqualified_extension(&["k"])?, "a")?;
    let plan = extension_capture_semi_join(build, "a.k")?;

    assert_captured_correlation_refused_by(
        &extension_unparser("safe", Some("t")),
        &plan,
        "an alias over an extension must not be taken as the whole emitted FROM",
    );
    Ok(())
}

/// The same for a probe side: an outer reference passes through the probe's own
/// `FROM` on its way out, so a probe whose emitted relation cannot be read
/// captures it just as a build side would.
///
/// The reference here names `x`, which neither side's plan holds, so nothing the
/// walk *can* read would refuse it — only the extension's unreadable `FROM "x"`
/// does. Asking the probe scope at all needs a `join.filter`, which is where an
/// [`Expr::OuterReferenceColumn`] reaches past the join.
///
/// [`Expr::OuterReferenceColumn`]: datafusion_expr::Expr::OuterReferenceColumn
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_captured_by_extension_probe()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = LogicalPlanBuilder::from(unqualified_extension(&["c", "d"])?)
        .project(vec![col("c"), col("d")])?
        .build()?;
    let build = table_scan(Some("b"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![out_ref_col(DataType::Int32, "x.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused_by(
        &extension_unparser("x", None),
        &plan,
        "an outer reference passing through an unreadable probe FROM must be refused",
    );
    Ok(())
}

/// The same rename with nothing stacked on it is folded away, and then the name
/// is exposed nowhere at all — so both halves of the correlation escape outward.
///
/// `b` holds `x` and `y`; the build side renames `x` to `c`, and with no operator
/// above it the emitter folds that projection into the `SELECT 1`. The body is
/// `FROM "b"`, which answers to `x` and `y` and to no `c` whatever. Un-guarded
/// the predicate is `("c" = "c")` with *both* halves binding to the outer
/// `"p"."c"`: the build half was meant to be `b.x`, so the correlation is gone,
/// `EXISTS` asks only whether `b` has a row, and the semi join keeps every probe
/// row.
///
/// The derived-table case above does not cover this one. There the emitted body
/// really does expose the renamed name, and a check that collected the output
/// names only when the emitter wraps the body would still refuse it — while
/// emitting the tautology here. Both arms are why those names are collected
/// whether or not the body is wrapped.
#[test]
fn test_unparse_left_semi_join_refuses_unqualified_capture_by_folded_rename() -> Result<()>
{
    let probe = unqualified_probe()?;
    let build_schema = int32_schema(&["x", "y"]);
    let build = table_scan(Some("b"), &build_schema, Some(vec![0]))?
        .project(vec![col("b.x").alias("c")])?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["c"], vec!["c"]),
            None,
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a rename the emitter folds away must still be seen to capture",
    );
    Ok(())
}

/// An outer reference only the build side answers to is captured, though the
/// probe knows nothing about it.
///
/// A `join.filter` is asked against both scopes because an `Expr::Column` in one
/// can only mean one of the join's own inputs, so a qualifier a single input owns
/// is attributable. An `Expr::OuterReferenceColumn` is not: it reaches past this
/// join to an enclosing query, and nothing about the probe can make it local.
/// Here the probe is `p` and the build is `b`, which share nothing — so asking
/// the probe as well answered "not captured" and emitted
/// `SELECT "p"."d" FROM "p" WHERE EXISTS (SELECT 1 FROM "b" WHERE ("b"."c" > 0))`,
/// where `"b"."c"` binds to the body's own `b`. The reference to the enclosing
/// query is gone, `EXISTS` asks only whether `b` has a positive row, and the semi
/// join keeps every probe row or none.
///
/// [`test_unparse_left_semi_join_refuses_shadowed_outer_reference`] covers the
/// case where both sides answer to the qualifier, which the two-scope test
/// already caught; this is the half it could not.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_only_the_build_answers_to()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("p"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("b"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("p.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![out_ref_col(DataType::Int32, "b.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference the build side alone answers to must be seen to capture",
    );
    Ok(())
}

/// The probe's own scope can shadow an outer reference just as the build's can.
///
/// An `OuterReferenceColumn` reaches past this join to an enclosing query, so on
/// its way out it passes through *both* emitted scopes: the `EXISTS` body's
/// `FROM`, and then the enclosing `SELECT`'s. Checking only the build side left
/// the second one open — here the probe is `b`, the build is `c`, and
/// `SELECT "b"."d" FROM "b" WHERE EXISTS (SELECT 1 FROM "c" WHERE ("b"."c" > 0))`
/// was emitted, where `"b"."c"` resolves at the probe's `b` and never reaches the
/// `b` the reference was written against.
///
/// This is the other half of
/// [`test_unparse_left_semi_join_refuses_outer_reference_only_the_build_answers_to`]:
/// neither scope excuses the other, so either one answering is a capture.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_only_the_probe_answers_to()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("b"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("c"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("b.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![out_ref_col(DataType::Int32, "b.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference the probe side alone answers to must be seen to capture",
    );
    Ok(())
}

/// A relation whose name folds onto an alias the unparser invents is captured by
/// that alias.
///
/// The build side is wrapped, so the body reads `FROM (...) AS derived_projection`
/// — a name that appears nowhere in the plan, which is why it is recognised by
/// spelling rather than found by the scope walk. Recognising it as written let a
/// probe relation named `DERIVED_PROJECTION` through, and a dialect that emits
/// both unquoted binds them together:
/// `... EXISTS (SELECT 1 FROM (SELECT b.c AS c FROM b) AS derived_projection WHERE (DERIVED_PROJECTION.c > 0))`,
/// where the correlated reference resolves at the invented derived table.
///
/// The invented-alias test therefore has to key its comparison the same way the
/// scope's own qualifiers are keyed; comparing one in emitted form and the other
/// as written is what left the gap.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_case_folded_invented_alias()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some(r#""DERIVED_PROJECTION""#), &schema, Some(vec![0, 1]))?
        .build()?;
    let build = table_scan(Some("b"), &schema, Some(vec![0]))?
        .project(vec![col("b.c").alias("c")])?
        .project(vec![col("c")])?
        .distinct()?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col(r#""DERIVED_PROJECTION".d"#)])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col(r#""DERIVED_PROJECTION".c"#).gt(lit(0))],
        )?
        .build()?;

    let dialect = CustomDialectBuilder::new()
        .with_requires_derived_table_alias(true)
        .build();
    assert_captured_correlation_refused_by(
        &Unparser::new(&dialect),
        &plan,
        "a relation folding onto an alias the unparser invents must be seen to capture",
    );
    Ok(())
}

/// Two relation names this dialect emits unquoted are one identifier by the time
/// they bind, however the plan spells them.
///
/// The probe is `T` and the build is `t`. A dialect with no identifier quoting
/// writes both bare, and an engine reading the statement case-folds an unquoted
/// identifier — so the body's `FROM t` answers to the outer `T` as well.
/// Comparing the qualifiers as written emitted
/// `SELECT T.d FROM T WHERE EXISTS (SELECT 1 FROM t WHERE (T.c > 0))`, whose
/// `T.c` binds inside: the correlation is gone and the semi join keeps every
/// probe row.
/// Builds `LeftSemi Join: "T".c > 0` where the probe is `T` and the build is `t`
/// — two relations that differ only in case, which is the axis the pair of tests
/// around it varies against the dialect.
fn case_distinct_relations_semi_join() -> Result<LogicalPlan> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some(r#""T""#), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![0]))?.build()?;
    LogicalPlanBuilder::from(probe)
        .project(vec![col(r#""T".d"#)])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col(r#""T".c"#).gt(lit(0))],
        )?
        .build()
}

#[test]
fn test_unparse_left_semi_join_refuses_capture_by_case_folded_unquoted_relation()
-> Result<()> {
    let plan = case_distinct_relations_semi_join()?;

    let dialect = CustomDialectBuilder::new().build();
    assert_captured_correlation_refused_by(
        &Unparser::new(&dialect),
        &plan,
        "two relations this dialect emits unquoted must be seen to collide",
    );
    Ok(())
}

/// The same two relations are refused on a quoting dialect too, and that is a
/// deliberate over-refusal rather than the same finding twice.
///
/// On PostgreSQL `"T"` and `"t"` really are distinct, the body's `FROM "t"` does
/// not answer to `"T"`, and
/// `SELECT "T"."d" FROM "T" WHERE EXISTS (SELECT 1 FROM "t" WHERE ("T"."c" > 0))`
/// binds exactly as written — so refusing costs a pushdown that did not need to
/// be spent.
///
/// It is spent because quoting does not say how the engine compares what was
/// written: DuckDB matches identifiers case-insensitively even quoted, and
/// BigQuery does so for column names while always emitting backticks. Keying on
/// the quote style would therefore keep this pushdown and go on emitting the
/// capture on those dialects, which is the trade the wrong way round. Asking the
/// dialect properly is spiceai/spiceai#13474; until then the fold is
/// unconditional, and this test is what records the price.
#[test]
fn test_unparse_left_semi_join_refuses_case_distinct_quoted_relations() -> Result<()> {
    let plan = case_distinct_relations_semi_join()?;

    assert_captured_correlation_refused_by(
        &Unparser::new(&UnparserPostgreSqlDialect {}),
        &plan,
        "the fold is unconditional, so a quoting dialect is refused as well",
    );
    Ok(())
}

/// A column name the dialect rewrites is compared as it is written, not as the
/// plan holds it.
///
/// `col_to_sql` passes every column name through `Dialect::col_alias_overrides`,
/// and BigQuery rewrites `min(a)` — not a legal identifier there — to
/// `min_40a_41`. The build relation `b` really has a column of that name, so the
/// emitted unqualified reference binds inside the body. Comparing the plan's
/// `min(a)` against the exposed `min_40a_41` reported no collision and emitted
/// ``SELECT ... FROM `p` WHERE EXISTS (SELECT 1 FROM `b` WHERE (`min_40a_41` = `b`.`y`))``,
/// where the correlation is lost and the semi join keeps every probe row.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_dialect_rewritten_column_name()
-> Result<()> {
    let probe = table_scan(Some("p"), &int32_schema(&["min(a)", "d"]), Some(vec![0, 1]))?
        .project(vec![
            col(r#"p."min(a)""#).alias("min(a)"),
            col("p.d").alias("d"),
        ])?
        .build()?;
    let build = table_scan(
        Some("b"),
        &int32_schema(&["min_40a_41", "y"]),
        Some(vec![0, 1]),
    )?
    .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["min(a)"], vec!["b.y"]),
            None,
        )?
        .build()?;

    assert_captured_correlation_refused_by(
        &Unparser::new(&BigQueryDialect {}),
        &plan,
        "a column name the dialect rewrites must be compared in its rewritten form",
    );
    Ok(())
}

/// The same rewrite reaches the body's exposed names, not just the reference.
///
/// Here the build side renames a column to `min(a)` and stacks enough operators
/// that the emitter wraps it, so the derived table carries that column under the
/// name the dialect writes — `min_40a_41`, the same form the probe's reference
/// takes. Comparing the reference only against the columns' *plan* names misses
/// it and emits
/// ``... EXISTS (SELECT 1 FROM (SELECT `b`.`x` AS `min_40a_41` FROM `b`) WHERE (`min_40a_41` = `min_40a_41`))``,
/// both halves binding to the derived table: the inner tautology again.
///
/// This is why each exposed column is tested under both names — its own, which a
/// relation emitted bare answers to, and the rewritten one a derived table
/// carries it under. Which of the two the body will be is not known where the
/// comparison happens.
#[test]
fn test_unparse_left_semi_join_refuses_rewritten_capture_by_derived_table_column()
-> Result<()> {
    let probe = table_scan(Some("p"), &int32_schema(&["min(a)", "d"]), Some(vec![0, 1]))?
        .project(vec![
            col(r#"p."min(a)""#).alias("min(a)"),
            col("p.d").alias("d"),
        ])?
        .build()?;
    let build = table_scan(Some("b"), &int32_schema(&["x", "y"]), Some(vec![0]))?
        .project(vec![col("b.x").alias("min(a)")])?
        .project(vec![col(r#""min(a)""#)])?
        .distinct()?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["min(a)"], vec!["min(a)"]),
            None,
        )?
        .build()?;

    assert_captured_correlation_refused_by(
        &Unparser::new(&BigQueryDialect {}),
        &plan,
        "a derived table carrying a column under its rewritten name must be seen to capture",
    );
    Ok(())
}

/// A relation the build side renames is not in the emitted scope under its own
/// name, so a correlation naming it is not captured and must keep its pushdown.
///
/// The build side scans `t` — the probe's own relation — but emits it as
/// `FROM "t" AS "derived"`. An alias *replaces* the name it is given to, so `"t"`
/// is not addressable inside the subquery and `"t"."c" > 0` binds outward,
/// exactly as intended.
///
/// This is why the scope walk stops at a `SubqueryAlias` rather than descending:
/// collecting the renamed `t` would read this as a collision and refuse working
/// SQL.
#[test]
fn test_unparse_left_semi_join_keeps_relation_enclosed_by_build_side_alias() -> Result<()>
{
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![0]))?
        .alias("derived")?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t.c").gt(lit(0))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t"."d" FROM "t" WHERE EXISTS (SELECT 1 FROM "t" AS "derived" WHERE ("t"."c" > 0))"#
    );
    Ok(())
}

/// A build side aliased to the probe's own name captures through the alias.
///
/// The enclosed relation is `t2`, which shares nothing with the probe — what
/// collides is the alias the derived table is given. `EXISTS (SELECT 1 FROM
/// (...) AS "t" WHERE ...)` puts `"t"` in the inner scope, so a correlated
/// reference qualified by `t` binds there.
///
/// This is the half of the scope the walk itself supplies: reaching a
/// `SubqueryAlias`, it records that alias as a qualifier and stops there rather
/// than descending to the relation the alias replaces.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_build_side_alias() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t2"), &schema, Some(vec![0]))?
        .alias("t")?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a build side aliased to the probe's name must be seen to capture",
    );
    Ok(())
}

/// A build side that projects away every column still names its relation in the
/// emitted `FROM`, so it still captures.
///
/// `TableScan t` with `projection=[]` has no qualified output field at all, so
/// reading the scope off the output schema reports it as sharing nothing. The SQL
/// says otherwise: `FROM "t"` is emitted, and a probe-only filter `"t"."c" > 0`
/// binds to that inner `t`. The `EXISTS` then asks whether *any* inner row is
/// positive rather than testing each probe row — one answer for the whole join,
/// so a semi join keeps every probe row or none.
///
/// Before the scope was read from the relations the `FROM` introduces, this
/// unparsed to
/// `SELECT "t"."d" FROM "t" WHERE EXISTS (SELECT 1 FROM "t" WHERE ("t"."c" > 0))`.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_unprojected_build_relation()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a build relation with no projected columns must still be seen to capture",
    );
    Ok(())
}

/// A dialect that spells columns in full keeps those two relations apart, so
/// there is no capture and the pushdown must survive.
///
/// This is the other half of the finding above: keying the refusal on the bare
/// table name unconditionally refuses this plan, whose SQL binds correctly —
/// `public.t.c` names the outer relation and nothing shadows it. The refusal has
/// to ask the dialect how the qualifier will be spelled, which is why it goes
/// through `emitted_qualifier` rather than reading the `TableReference`.
///
/// The bound is absent deliberately: `exists_scope_name` is consulted only when
/// one is present, so nothing else in this file would refuse this shape, and a
/// guard keyed on the bare name would be the only thing standing between a
/// working pushdown and a query failure.
#[test]
fn test_unparse_left_semi_join_keeps_cross_schema_on_full_column_dialect() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("public.t"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("other.t"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("public.t.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("public.t.c").eq(col("other.t.c"))],
        )?
        .build()?;

    let dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .build();
    assert_snapshot!(
        Unparser::new(&dialect).plan_to_sql(&plan)?,
        @"SELECT public.t.d FROM public.t WHERE EXISTS (SELECT 1 FROM other.t WHERE (public.t.c = other.t.c))"
    );
    Ok(())
}

/// `exists_scope_name`'s probe-qualified arm still has a shape that reaches it,
/// and this pins that it does.
///
/// The correlation is carried entirely by a filter naming only probe relations,
/// and the two sides share no qualifier — so the capture guard returns without
/// refusing, and the bound then demands a scope name that has no build-side
/// relation to take. Without this, that arm has no coverage: the test that used
/// to reach it now stops at the capture guard instead, which is a refusal moving
/// rather than a shape disappearing.
#[test]
fn test_unparse_left_semi_join_refuses_probe_only_filter_with_bound() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan_with_filter_and_fetch(
        Some("t2"),
        &schema,
        Some(vec![0]),
        vec![],
        Some(5),
    )?
    .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.c").gt(col("t1.d"))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let err = unparser
        .plan_to_sql(&plan)
        .expect_err("a bound with no build-side qualifier to scope must be refused");
    assert_snapshot!(
        err,
        @"This feature is not implemented: Unparsing a row bound on an EXISTS-style join's build side is not supported when the correlation's only qualifier is one the probe side also answers to"
    );
    Ok(())
}

/// Two relations in different schemas that share a table name collide in the
/// emitted SQL even though their `TableReference`s differ.
///
/// On a dialect that does not spell columns in full, a qualified column renders
/// as its relation's last component, so both `public.t.c` and `other.t.c` emit as
/// `"t"."c"` — and inside `FROM "other"."t"` both bind to the inner relation, which
/// is the same tautology as the bare self-join. Comparing the references whole
/// would read these as disjoint and let it through; before this was caught, the
/// plan unparsed to
/// `SELECT "d" FROM "public"."t" WHERE EXISTS (SELECT 1 FROM "other"."t" WHERE ("t"."c" = "t"."c"))`.
#[test]
fn test_unparse_left_semi_join_refuses_cross_schema_name_collision() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("public.t"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("other.t"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("public.t.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("public.t.c").eq(col("other.t.c"))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "relations sharing an emitted name must be refused",
    );
    Ok(())
}

/// A build-side join key naming the shared relation is not a capture, and must
/// keep its pushdown.
///
/// The probe is `t1 INNER JOIN t` and the build side is `t`, so the two share the
/// name `t`, but the key is `(t1.c, t.c)`: only `t1.c` is correlated, and the
/// build half `t.c` is meant to bind to the subquery's own `t` — which is exactly
/// what it does. Refusing here would cost the pushdown on correct SQL, and
/// testing both halves of each pair rather than the correlated one does refuse
/// it, which is what this pins.
#[test]
fn test_unparse_left_semi_join_keeps_build_side_key_on_shared_relation() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?
        .join_on(
            table_scan(Some("t"), &schema, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            vec![col("t1.d").eq(col("t.c"))],
        )?
        .build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.d"), col("t.c")])?
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["t1.c"], vec!["t.c"]),
            None,
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d", "t"."c" FROM "t1" INNER JOIN "t" ON ("t1"."d" = "t"."c") WHERE EXISTS (SELECT 1 FROM "t" WHERE ("t1"."c" = "t"."c"))"#
    );
    Ok(())
}

/// The same capture through `join.on` rather than `join.filter`.
///
/// A shared-qualifier equality cannot be attributed to two sides, so
/// [`LogicalPlanBuilder::join_on`] puts it in the filter and the other refusals
/// here reach it there. Naming the join keys directly puts the identical
/// predicate in `on` instead, which is a plan the unparser can be handed even
/// though the higher-level builder will not produce it — and the emitted SQL,
/// and so the capture, is the same either way.
#[test]
fn test_unparse_left_semi_join_refuses_shadowed_correlation_in_join_keys() -> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![0]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t.d")])?
        .join(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec!["t.c"], vec!["t.c"]),
            None,
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a shadowed correlation in the join keys must be refused too",
    );
    Ok(())
}

/// Sharing a qualifier is not on its own a reason to refuse. Both sides here
/// answer to `t`, but the correlated half of the `on` pair names `t1`, which the
/// build side does not answer to, so nothing is captured and the plan unparses.
///
/// This is the boundary the refusal is drawn on: the correlated reference's own
/// qualifier, not an overlap between the two sides' relations. Refusing on
/// overlap alone would cost the pushdown on federated self-joins that unparse
/// correctly today — the same trade
/// [`test_unparse_left_semi_join_without_fetch_keeps_multi_relation_correlation`]
/// guards from the other direction.
#[test]
fn test_unparse_left_semi_join_keeps_overlapping_but_unreferenced_qualifier() -> Result<()>
{
    let plan = overlapping_but_unreferenced_qualifier_semi_join()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d", "t"."c" FROM "t1" INNER JOIN "t" ON ("t1"."c" = "t"."c") WHERE EXISTS (SELECT 1 FROM "t" INNER JOIN "t3" ON ("t"."c" = "t3"."c") WHERE ("t1"."c" = "t3"."c"))"#
    );
    Ok(())
}

/// An unbounded build side keeps the flat body: nothing decides which rows
/// survive, so the correlation can stay in the body's own `WHERE` and the extra
/// scope would be noise. Pins that the fix costs nothing when it is not needed.
#[test]
fn test_unparse_left_semi_join_without_fetch_stays_flat() -> Result<()> {
    let plan =
        exists_join_with_build_side_fetch(datafusion_expr::JoinType::LeftSemi, None)?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "t1"."d" FROM "t1" WHERE EXISTS (SELECT 1 FROM "t2" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

#[test]
fn test_unparse_left_mark_join() -> Result<()> {
    // select t1.d from t1 where t1.d < 0 OR exists (select 1 from t2 where t1.c = t2.c)
    let schema = Schema::new(vec![
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ]);
    // Filter: __correlated_sq_1.mark OR t1.d < Int32(0)
    //   Projection: t1.d
    //     LeftMark Join:  Filter: t1.c = __correlated_sq_1.c
    //       TableScan: t1 projection=[c, d]
    //       SubqueryAlias: __correlated_sq_1
    //         TableScan: t2 projection=[c]
    let table_scan1 = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let table_scan2 = table_scan(Some("t2"), &schema, Some(vec![0]))?.build()?;
    let subquery = subquery_alias(table_scan2, "__correlated_sq_1")?;
    let plan = LogicalPlanBuilder::from(table_scan1)
        .join_on(
            subquery,
            datafusion_expr::JoinType::LeftMark,
            vec![col("t1.c").eq(col("__correlated_sq_1.c"))],
        )?
        .project(vec![col("t1.d")])?
        .filter(col("mark").or(col("t1.d").lt(lit(0))))?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t1"."d" FROM "t1" WHERE (EXISTS (SELECT 1 FROM "t2" AS "__correlated_sq_1" WHERE ("t1"."c" = "__correlated_sq_1"."c")) OR ("t1"."d" < 0))"#
    );
    Ok(())
}

#[test]
fn test_unparse_right_semi_join() -> Result<()> {
    // select t2.c, t2.d from t1 right semi join t2 on t1.c = t2.c where t2.c <= 1
    let schema = Schema::new(vec![
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ]);
    // Filter: t2.c <= Int64(1)
    //   RightSemi Join: t1.c = t2.c
    //     TableScan: t1 projection=[c, d]
    //     Projection: t2.c, t2.d
    //       TableScan: t2 projection=[c, d]
    let left = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let right_table_scan = table_scan(Some("t2"), &schema, Some(vec![0, 1]))?.build()?;
    let right = LogicalPlanBuilder::from(right_table_scan)
        .project(vec![col("c"), col("d")])?
        .build()?;
    let plan = LogicalPlanBuilder::from(left)
        .join(
            right,
            datafusion_expr::JoinType::RightSemi,
            (
                vec![Column::from_qualified_name("t1.c")],
                vec![Column::from_qualified_name("t2.c")],
            ),
            None,
        )?
        .filter(col("t2.c").lt_eq(lit(1i64)))?
        .build()?;
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t2"."c", "t2"."d" FROM "t2" WHERE ("t2"."c" <= 1) AND EXISTS (SELECT 1 FROM "t1" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

#[test]
fn test_unparse_right_anti_join() -> Result<()> {
    // select t2.c, t2.d from t1 right anti join t2 on t1.c = t2.c where t2.c <= 1
    let schema = Schema::new(vec![
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ]);
    // Filter: t2.c <= Int64(1)
    //   RightAnti Join: t1.c = t2.c
    //     TableScan: t1 projection=[c, d]
    //     Projection: t2.c, t2.d
    //       TableScan: t2 projection=[c, d]
    let left = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?.build()?;
    let right_table_scan = table_scan(Some("t2"), &schema, Some(vec![0, 1]))?.build()?;
    let right = LogicalPlanBuilder::from(right_table_scan)
        .project(vec![col("c"), col("d")])?
        .build()?;
    let plan = LogicalPlanBuilder::from(left)
        .join(
            right,
            datafusion_expr::JoinType::RightAnti,
            (
                vec![Column::from_qualified_name("t1.c")],
                vec![Column::from_qualified_name("t2.c")],
            ),
            None,
        )?
        .filter(col("t2.c").lt_eq(lit(1i64)))?
        .build()?;
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t2"."c", "t2"."d" FROM "t2" WHERE ("t2"."c" <= 1) AND NOT EXISTS (SELECT 1 FROM "t1" WHERE ("t1"."c" = "t2"."c"))"#
    );
    Ok(())
}

#[test]
fn test_unparse_cross_join_with_table_scan_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    // Cross Join:
    //   SubqueryAlias: t1
    //     TableScan: test projection=[v]
    //   SubqueryAlias: t2
    //     TableScan: test projection=[v]
    let table_scan1 = table_scan(Some("test"), &schema, Some(vec![1]))?.build()?;
    let table_scan2 = table_scan(Some("test"), &schema, Some(vec![1]))?.build()?;
    let plan = LogicalPlanBuilder::from(subquery_alias(table_scan1, "t1")?)
        .cross_join(subquery_alias(table_scan2, "t2")?)?
        .build()?;
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t1"."v", "t2"."v" FROM "test" AS "t1" CROSS JOIN "test" AS "t2""#
    );
    Ok(())
}

#[test]
fn test_unparse_inner_join_with_table_scan_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    // Inner Join:
    //   SubqueryAlias: t1
    //     TableScan: test projection=[v]
    //   SubqueryAlias: t2
    //     TableScan: test projection=[v]
    let table_scan1 = table_scan(Some("test"), &schema, Some(vec![1]))?.build()?;
    let table_scan2 = table_scan(Some("test"), &schema, Some(vec![1]))?.build()?;
    let plan = LogicalPlanBuilder::from(subquery_alias(table_scan1, "t1")?)
        .join_on(
            subquery_alias(table_scan2, "t2")?,
            datafusion_expr::JoinType::Inner,
            vec![col("t1.v").eq(col("t2.v"))],
        )?
        .build()?;
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t1"."v", "t2"."v" FROM "test" AS "t1" INNER JOIN "test" AS "t2" ON ("t1"."v" = "t2"."v")"#
    );
    Ok(())
}

#[test]
fn test_unparse_left_semi_join_with_table_scan_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    // LeftSemi Join:
    //   SubqueryAlias: t1
    //     TableScan: test projection=[v]
    //   SubqueryAlias: t2
    //     TableScan: test projection=[v]
    let table_scan1 = table_scan(Some("test"), &schema, Some(vec![1]))?.build()?;
    let table_scan2 = table_scan(Some("test"), &schema, Some(vec![1]))?.build()?;
    let plan = LogicalPlanBuilder::from(subquery_alias(table_scan1, "t1")?)
        .join_on(
            subquery_alias(table_scan2, "t2")?,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t1.v").eq(col("t2.v"))],
        )?
        .build()?;
    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "t1"."v" FROM "test" AS "t1" WHERE EXISTS (SELECT 1 FROM "test" AS "t2" WHERE ("t1"."v" = "t2"."v"))"#
    );
    Ok(())
}

#[test]
fn test_unparse_window() -> Result<()> {
    // SubqueryAlias: t
    // Projection: t.k, t.v, rank() PARTITION BY [t.k] ORDER BY [t.v ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW AS r
    //     Filter: rank() PARTITION BY [t.k] ORDER BY [t.v ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW = UInt64(1)
    //     WindowAggr: windowExpr=[[rank() PARTITION BY [t.k] ORDER BY [t.v ASC NULLS LAST] RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW]]
    //         TableScan: t projection=[k, v]

    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(rank_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![col("k")],
            order_by: vec![col("v").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }));
    let table = table_scan(Some("test"), &schema, Some(vec![0, 1]))?.build()?;
    let plan = LogicalPlanBuilder::window_plan(table, vec![window_expr.clone()])?;

    let name = plan.schema().fields().last().unwrap().name().clone();
    let plan = LogicalPlanBuilder::from(plan)
        .filter(col(name.clone()).eq(lit(1i64)))?
        .project(vec![col("k"), col("v"), col(name)])?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "test"."k", "test"."v", "rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING" FROM (SELECT "test"."k" AS "k", "test"."v" AS "v", rank() OVER (PARTITION BY "test"."k" ORDER BY "test"."v" ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING" FROM "test") AS "test" WHERE ("rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING" = 1)"#
    );

    let unparser = Unparser::new(&UnparserMySqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT `test`.`k`, `test`.`v`, `rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` FROM (SELECT `test`.`k` AS `k`, `test`.`v` AS `v`, rank() OVER (PARTITION BY `test`.`k` ORDER BY `test`.`v` ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` FROM `test`) AS `test` WHERE (`rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` = 1)"
    );

    let unparser = Unparser::new(&SqliteDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT `test`.`k`, `test`.`v`, `rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` FROM (SELECT `test`.`k` AS `k`, `test`.`v` AS `v`, rank() OVER (PARTITION BY `test`.`k` ORDER BY `test`.`v` ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` FROM `test`) AS `test` WHERE (`rank() PARTITION BY [test.k] ORDER BY [test.v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` = 1)"
    );

    let unparser = Unparser::new(&DefaultDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT test.k, test.v, rank() OVER (PARTITION BY test.k ORDER BY test.v ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM test QUALIFY (rank() OVER (PARTITION BY test.k ORDER BY test.v ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) = 1)"
    );

    // without table qualifier
    let table = table_scan(Some("test"), &schema, Some(vec![0, 1]))?.build()?;
    let table = LogicalPlanBuilder::from(table)
        .project(vec![col("k").alias("k"), col("v").alias("v")])?
        .build()?;
    let plan = LogicalPlanBuilder::window_plan(table, vec![window_expr])?;

    let name = plan.schema().fields().last().unwrap().name().clone();
    let plan = LogicalPlanBuilder::from(plan)
        .filter(col(name.clone()).eq(lit(1i64)))?
        .project(vec![col("k"), col("v"), col(name)])?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT "k", "v", "rank() PARTITION BY [k] ORDER BY [v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING" FROM (SELECT "k" AS "k", "v" AS "v", rank() OVER (PARTITION BY "k" ORDER BY "v" ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "rank() PARTITION BY [k] ORDER BY [v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING" FROM (SELECT "test"."k" AS "k", "test"."v" AS "v" FROM "test") AS "derived_projection") AS "__qualify_subquery" WHERE ("rank() PARTITION BY [k] ORDER BY [v ASC NULLS FIRST] ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING" = 1)"#
    );

    Ok(())
}

#[test]
fn test_unparse_window_over_aggregate_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![col("time").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("gas"), &schema, None)?
        .aggregate(vec![col("time")], vec![sum(col("value")).alias("sum_n")])?
        .window(vec![window_expr])?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT sum(gas."value") AS sum_n, gas."time", row_number() OVER (ORDER BY gas."time" ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM gas GROUP BY gas."time""#
    );

    Ok(())
}

#[test]
fn test_unparse_filter_on_window_over_aggregate_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![col("time").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("gas"), &schema, None)?
        .aggregate(vec![col("time")], vec![sum(col("value")).alias("sum_n")])?
        .window(vec![window_expr])?
        .filter(col("row_idx").eq(lit(0i64)))?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT sum(gas."value") AS sum_n, gas."time", row_number() OVER (ORDER BY gas."time" ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM gas GROUP BY gas."time" QUALIFY (row_number() OVER (ORDER BY gas."time" ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) = 0)"#
    );

    Ok(())
}

#[test]
fn test_unparse_filter_on_aggregate_output_above_window_without_projection() -> Result<()>
{
    let schema = Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![col("time").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("gas"), &schema, None)?
        .aggregate(vec![col("time")], vec![sum(col("value")).alias("sum_n")])?
        .window(vec![window_expr])?
        .filter(col("sum_n").eq(lit(0f64)))?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT sum(gas."value") AS sum_n, gas."time", row_number() OVER (ORDER BY gas."time" ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM gas GROUP BY gas."time" QUALIFY (sum(gas."value") = 0.0)"#
    );

    Ok(())
}

#[test]
fn test_unparse_window_over_table_scan_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![col("k")],
            order_by: vec![col("v").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("test"), &schema, None)?
        .window(vec![window_expr])?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT *, row_number() OVER (PARTITION BY test.k ORDER BY test.v ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM test"
    );

    Ok(())
}

#[test]
fn test_unparse_stacked_windows_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    let row_number_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![col("k")],
            order_by: vec![col("v").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let rank_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(rank_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![col("v").sort(false, false)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("rank_idx");
    let plan = table_scan(Some("test"), &schema, None)?
        .window(vec![row_number_expr])?
        .window(vec![rank_expr])?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT *, row_number() OVER (PARTITION BY test.k ORDER BY test.v ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx, rank() OVER (ORDER BY test.v DESC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS rank_idx FROM test"
    );

    Ok(())
}

#[test]
fn test_unparse_window_over_distinct_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![col("v").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("test"), &schema, None)?
        .distinct()?
        .window(vec![window_expr])?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT *, row_number() OVER (ORDER BY derived_window_input.v ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM (SELECT DISTINCT * FROM test) AS derived_window_input"
    );

    Ok(())
}

#[test]
fn test_unparse_window_over_limit_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![col("v").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("test"), &schema, None)?
        .limit(0, Some(10))?
        .window(vec![window_expr])?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT *, row_number() OVER (ORDER BY derived_window_input.v ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM (SELECT * FROM test LIMIT 10) AS derived_window_input"
    );

    Ok(())
}

#[test]
fn test_unparse_window_over_projection_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![col("v_alias").sort(true, true)],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("test"), &schema, None)?
        .project(vec![col("k"), col("v").alias("v_alias")])?
        .window(vec![window_expr])?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @"SELECT *, row_number() OVER (ORDER BY derived_window_input.v_alias ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM (SELECT test.k, test.v AS v_alias FROM test) AS derived_window_input"
    );

    Ok(())
}

#[test]
fn test_unparse_window_over_derived_aggregate_without_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ]);
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(row_number_udwf()),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: vec![],
            order_by: vec![
                Expr::Column(Column::new(Some(TableReference::bare("agg")), "sum_n"))
                    .sort(true, true),
            ],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }))
    .alias("row_idx");
    let plan = table_scan(Some("gas"), &schema, None)?
        .aggregate(vec![col("time")], vec![sum(col("value")).alias("sum_n")])?
        .alias("agg")?
        .window(vec![window_expr])?
        .build()?;

    let sql = Unparser::default().plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT *, row_number() OVER (ORDER BY agg.sum_n ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS row_idx FROM (SELECT sum(gas."value") AS sum_n, gas."time" FROM gas GROUP BY gas."time") AS agg"#
    );

    Ok(())
}

#[test]
fn test_array_to_sql_postgres() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT [1, 2, 3, 4, 5]",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserPostgreSqlDialect {},
        expected: @"SELECT ARRAY[1, 2, 3, 4, 5]",
    );
    Ok(())
}

#[test]
fn test_like_filter() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT first_name FROM person WHERE first_name LIKE '%John%'"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.first_name FROM person WHERE person.first_name LIKE '%John%'"
    );
}

#[test]
fn test_ilike_filter() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT first_name FROM person WHERE first_name ILIKE '%john%'"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.first_name FROM person WHERE person.first_name ILIKE '%john%'"
    );
}

#[test]
fn test_not_like_filter() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT first_name FROM person WHERE first_name NOT LIKE 'A%'"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.first_name FROM person WHERE person.first_name NOT LIKE 'A%'"
    );
}

#[test]
fn test_not_ilike_filter() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT first_name FROM person WHERE first_name NOT ILIKE 'a%'"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.first_name FROM person WHERE person.first_name NOT ILIKE 'a%'"
    );
}

#[test]
fn test_like_filter_with_escape() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT first_name FROM person WHERE first_name LIKE 'A!_%' ESCAPE '!'"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.first_name FROM person WHERE person.first_name LIKE 'A!_%' ESCAPE '!'"
    );
}

#[test]
fn test_not_like_filter_with_escape() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT first_name FROM person WHERE first_name NOT LIKE 'A!_%' ESCAPE '!'"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.first_name FROM person WHERE person.first_name NOT LIKE 'A!_%' ESCAPE '!'"
    );
}

#[test]
fn test_not_ilike_filter_with_escape() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT first_name FROM person WHERE first_name NOT ILIKE 'A!_%' ESCAPE '!'"#,
    );
    assert_snapshot!(
        statement,
        @"SELECT person.first_name FROM person WHERE person.first_name NOT ILIKE 'A!_%' ESCAPE '!'"
    );
}

#[test]
fn test_struct_expr() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"WITH test AS (SELECT STRUCT(STRUCT('Product Name' as name) as product) AS metadata) SELECT metadata.product FROM test WHERE metadata.product.name  = 'Product Name'"#,
    );
    assert_snapshot!(
        statement,
        @r#"SELECT test."metadata".product FROM (SELECT {product: {"name": 'Product Name'}} AS "metadata") AS test WHERE (test."metadata".product."name" = 'Product Name')"#
    );

    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"WITH test AS (SELECT STRUCT(STRUCT('Product Name' as name) as product) AS metadata) SELECT metadata.product FROM test WHERE metadata['product']['name']  = 'Product Name'"#,
    );
    assert_snapshot!(
        statement,
        @r#"SELECT test."metadata".product FROM (SELECT {product: {"name": 'Product Name'}} AS "metadata") AS test WHERE (test."metadata".product."name" = 'Product Name')"#
    );
}

#[test]
fn test_struct_expr2() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT STRUCT(STRUCT('Product Name' as name) as product)['product']['name']  = 'Product Name';"#,
    );
    assert_snapshot!(
        statement,
        @r#"SELECT ({product: {"name": 'Product Name'}}.product."name" = 'Product Name')"#
    );
}

#[test]
fn test_struct_expr3() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"WITH
                test AS (
                    SELECT
                        STRUCT (
                            STRUCT (
                                STRUCT ('Product Name' as name) as product
                            ) AS metadata
                        ) AS c1
                )
            SELECT
                c1.metadata.product.name
            FROM
                test"#,
    );
    assert_snapshot!(
        statement,
        @r#"SELECT test.c1."metadata".product."name" FROM (SELECT {"metadata": {product: {"name": 'Product Name'}}} AS c1) AS test"#
    );
}

#[test]
fn test_json_access_1() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT j1_string:field FROM j1"#,
    );
    assert_snapshot!(
        statement,
        @r#"SELECT (j1.j1_string : 'field') FROM j1"#
    );
}

#[test]
fn test_json_access_2() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT j1_string:field[0] FROM j1"#,
    );
    assert_snapshot!(
        statement,
        @r#"SELECT (j1.j1_string : 'field[0]') FROM j1"#
    );
}

#[test]
fn test_json_access_3() {
    let statement = generate_round_trip_statement(
        GenericDialect {},
        r#"SELECT j1_string:field.inner1['inner2'] FROM j1"#,
    );
    assert_snapshot!(
        statement,
        @r#"SELECT (j1.j1_string : 'field.inner1[''inner2'']') FROM j1"#
    );
}

/// Roundtrip test for a subquery aggregate with column aliases.
/// Ensures that `subquery_alias_inner_query_and_columns` unwrapping
/// a Projection -> Aggregate still triggers the derived-subquery path.
#[test]
fn roundtrip_subquery_aggregate_with_column_alias() -> Result<(), DataFusionError> {
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM (SELECT max(j1_id) FROM j1) AS c(id)",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT c.id FROM (SELECT max(j1.j1_id) FROM j1) AS c (id)",
    );
    Ok(())
}

/// Test that unparsing a manually constructed join with a subquery aggregate
/// preserves the MAX aggregate function.
///
/// Builds the equivalent of:
///   SELECT j1.j1_string FROM j1
///     JOIN (SELECT max(j2_id) AS max_id FROM j2) AS b
///     ON j1.j1_id = b.max_id
#[test]
fn test_unparse_manual_join_with_subquery_aggregate() -> Result<()> {
    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let j1_schema = context
        .get_table_source(TableReference::bare("j1"))?
        .schema();
    let j2_schema = context
        .get_table_source(TableReference::bare("j2"))?
        .schema();

    // Build the right side: SELECT max(j2_id) AS max_id FROM j2
    let right_scan = table_scan(Some("j2"), &j2_schema, None)?.build()?;
    let right_agg = LogicalPlanBuilder::from(right_scan)
        .aggregate(
            vec![] as Vec<Expr>,
            vec![max(col("j2.j2_id")).alias("max_id")],
        )?
        .build()?;
    let right_subquery = subquery_alias(right_agg, "b")?;

    // Build the full plan: SELECT j1.j1_string FROM j1 JOIN (...) AS b ON j1.j1_id = b.max_id
    let left_scan = table_scan(Some("j1"), &j1_schema, None)?.build()?;
    let plan = LogicalPlanBuilder::from(left_scan)
        .join(
            right_subquery,
            datafusion_expr::JoinType::Inner,
            (
                vec![Column::from_qualified_name("j1.j1_id")],
                vec![Column::from_qualified_name("b.max_id")],
            ),
            None,
        )?
        .project(vec![col("j1.j1_string")])?
        .build()?;

    let unparser = Unparser::default();
    let sql = unparser.plan_to_sql(&plan)?.to_string();
    let sql_upper = sql.to_uppercase();
    assert!(
        sql_upper.contains("MAX("),
        "Unparsed SQL should preserve the MAX aggregate function call, got: {sql}"
    );

    Ok(())
}

/// Regression test for https://github.com/apache/datafusion/issues/21490
///
/// When the outer Projection excludes a Sort column whose definition only
/// exists as an alias in the inner Projection, the Unparser must inline the
/// underlying expression into ORDER BY rather than emitting the now-missing
/// alias name.
#[test]
fn test_sort_on_aliased_column_dropped_by_outer_projection() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("X", DataType::Utf8, true),
        Field::new("Y", DataType::Utf8, true),
        Field::new("Z", DataType::Utf8, true),
    ]);

    // Build:
    //   Projection: [a, b]                         -- outer: excludes sort column "c"
    //     Sort: [c DESC, fetch=1]                   -- references alias "c"
    //       Projection: [X AS a, Y AS b, Z AS c]    -- defines alias "c"
    //         SubqueryAlias: t
    //           TableScan: phys_table [X, Y, Z]
    let plan = table_scan(Some("phys_table"), &schema, None)?
        .alias("t")?
        .project(vec![
            Expr::Column(Column::new(Some(TableReference::bare("t")), "X")).alias("a"),
            Expr::Column(Column::new(Some(TableReference::bare("t")), "Y")).alias("b"),
            Expr::Column(Column::new(Some(TableReference::bare("t")), "Z")).alias("c"),
        ])?
        .sort_with_limit(
            vec![Expr::Column(Column::new_unqualified("c")).sort(false, true)],
            Some(1),
        )?
        .project(vec![
            Expr::Column(Column::new_unqualified("a")),
            Expr::Column(Column::new_unqualified("b")),
        ])?
        .build()?;

    let unparser = Unparser::default();
    let sql = unparser.plan_to_sql(&plan)?;

    // ORDER BY must reference the physical column, not the dropped alias.
    assert_snapshot!(
        sql,
        @r#"SELECT t."X" AS a, t."Y" AS b FROM phys_table AS t ORDER BY t."Z" DESC NULLS FIRST LIMIT 1"#
    );

    Ok(())
}

#[test]
fn snowflake_unnest_to_lateral_flatten_simple() -> Result<(), DataFusionError> {
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3])",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "_unnest_1"."VALUE" FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "_unnest_1""#,
    );
    Ok(())
}

#[test]
fn snowflake_unnest_to_lateral_flatten_with_cross_join() -> Result<(), DataFusionError> {
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]), j1",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "_unnest_1"."VALUE", "j1"."j1_id", "j1"."j1_string" FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "_unnest_1" CROSS JOIN "j1""#,
    );
    Ok(())
}

#[test]
fn snowflake_unnest_to_lateral_flatten_cross_join_inline() -> Result<(), DataFusionError>
{
    // Cross join with two inline UNNEST sources — both produce valid FLATTEN.
    // NOTE: UNNEST(table.column) is NOT tested with Snowflake because
    // LATERAL FLATTEN(INPUT => col) requires the column to be a Snowflake
    // VARIANT/ARRAY type, which cannot be validated at unparse time.
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) u(c1) JOIN j1 ON u.c1 = j1.j1_id",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        // NOTE: SELECT correctly uses VALUE, but the JOIN ON condition
        // still references the original column alias (c1) because join
        // filters are rendered outside reconstruct_select_statement.
        expected: @r#"SELECT "u"."VALUE", "j1"."j1_id", "j1"."j1_string" FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "u" INNER JOIN "j1" ON ("u"."c1" = "j1"."j1_id")"#,
    );
    Ok(())
}

// --- Edge case tests for Snowflake FLATTEN ---

#[test]
fn snowflake_flatten_implicit_from() -> Result<(), DataFusionError> {
    // UNNEST in SELECT clause (no explicit FROM UNNEST) — implicit table factor
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT UNNEST([1,2,3])",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "_unnest_1"."VALUE" FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "_unnest_1""#,
    );
    Ok(())
}

#[test]
fn snowflake_flatten_string_array() -> Result<(), DataFusionError> {
    // String array unnest
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST(['a','b','c'])",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "_unnest_1"."VALUE" FROM LATERAL FLATTEN(INPUT => ['a', 'b', 'c']) AS "_unnest_1""#,
    );
    Ok(())
}

#[test]
fn snowflake_flatten_select_unnest_with_alias() -> Result<(), DataFusionError> {
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT UNNEST([1,2,3]) as c1",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "_unnest_1"."VALUE" AS "c1" FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "_unnest_1""#,
    );
    Ok(())
}

#[test]
fn snowflake_flatten_select_unnest_plus_literal() -> Result<(), DataFusionError> {
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT UNNEST([1,2,3]), 1",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "_unnest_1"."VALUE", "Int64(1)" FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "_unnest_1""#,
    );
    Ok(())
}

#[test]
fn snowflake_flatten_from_unnest_with_table_alias() -> Result<(), DataFusionError> {
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM UNNEST([1,2,3]) AS t1 (c1)",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "t1"."VALUE" FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "t1""#,
    );
    Ok(())
}

#[test]
fn snowflake_flatten_unnest_from_subselect() -> Result<(), DataFusionError> {
    // UNNEST operating on an array column produced by a subselect.
    // Uses unnest_table which has array_col (List<Int64>).
    // The filter uses array_col IS NOT NULL — a simple predicate
    // that doesn't involve struct types (which Snowflake FLATTEN can't handle).
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT UNNEST(array_col) FROM (SELECT array_col FROM unnest_table WHERE array_col IS NOT NULL LIMIT 3)",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "_unnest_1"."VALUE" FROM (SELECT "unnest_table"."array_col" FROM "unnest_table" WHERE "unnest_table"."array_col" IS NOT NULL LIMIT 3) CROSS JOIN LATERAL FLATTEN(INPUT => "unnest_table"."array_col") AS "_unnest_1""#,
    );
    Ok(())
}

/// Dummy scalar UDF for testing — takes a string and returns List<Int64>.
/// Simulates any UDF that extracts an array from a column (e.g. parsing
/// JSON, splitting a delimited string, etc.).
#[derive(Debug, PartialEq, Eq, Hash)]
struct ExtractArrayUdf {
    signature: Signature,
}

impl ExtractArrayUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Utf8], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for ExtractArrayUdf {
    fn name(&self) -> &str {
        "extract_array"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new_list_field(
            DataType::Int64,
            true,
        ))))
    }
    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        unimplemented!("test stub")
    }
}

#[test]
fn snowflake_flatten_unnest_udf_result() -> Result<(), DataFusionError> {
    // UNNEST on a UDF result: extract_array(col) returns List<Int64>,
    // then UNNEST flattens it. This exercises the path where the FLATTEN
    // INPUT is a UDF call rather than a bare column reference.
    let sql = "SELECT UNNEST(extract_array(j1_string)) AS items FROM j1 LIMIT 5";

    let statement = Parser::new(&GenericDialect {})
        .try_with_sql(sql)?
        .parse_statement()?;

    let state = MockSessionState::default()
        .with_aggregate_function(max_udaf())
        .with_aggregate_function(min_udaf())
        .with_scalar_function(Arc::new(ScalarUDF::new_from_impl(ExtractArrayUdf::new())))
        .with_expr_planner(Arc::new(CoreFunctionPlanner::default()))
        .with_expr_planner(Arc::new(NestedFunctionPlanner))
        .with_expr_planner(Arc::new(FieldAccessPlanner));

    let context = MockContextProvider { state };
    let sql_to_rel = SqlToRel::new(&context);
    let plan = sql_to_rel
        .sql_statement_to_plan(statement)
        .unwrap_or_else(|e| panic!("Failed to parse sql: {sql}\n{e}"));

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    insta::assert_snapshot!(actual, @r#"SELECT "_unnest_1"."VALUE" AS "items" FROM "j1" CROSS JOIN LATERAL FLATTEN(INPUT => extract_array("j1"."j1_string")) AS "_unnest_1" LIMIT 5"#);
    Ok(())
}

#[test]
fn snowflake_flatten_limit_between_projection_and_unnest() -> Result<(), DataFusionError>
{
    // Build: Projection → Limit → Unnest → Projection → TableScan
    // The optimizer can insert a Limit between the outer Projection and the
    // Unnest. The FLATTEN code path must look through transparent nodes
    // (Limit, Sort) to find the Unnest.
    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .unnest_column("__unnest_placeholder(items)")?
        .limit(0, Some(5))? // Limit BETWEEN outer Projection and Unnest
        .project(vec![col("__unnest_placeholder(items)").alias("item")])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    // Must contain LATERAL FLATTEN — the Limit must not prevent FLATTEN detection
    insta::assert_snapshot!(actual, @r#"SELECT "_unnest_1"."VALUE" AS "item" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1" LIMIT 5"#);
    Ok(())
}

#[test]
fn snowflake_flatten_sort_between_projection_and_unnest() -> Result<(), DataFusionError> {
    // Build: Projection → Sort → Unnest → Projection → TableScan
    // Same as Limit test but with Sort instead.
    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .unnest_column("__unnest_placeholder(items)")?
        .sort(vec![col("__unnest_placeholder(items)").sort(true, true)])?
        .project(vec![col("__unnest_placeholder(items)").alias("item")])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    // Must contain LATERAL FLATTEN — the Sort must not prevent FLATTEN detection
    insta::assert_snapshot!(actual, @r#"SELECT "_unnest_1"."VALUE" AS "item" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1" ORDER BY "_unnest_1"."VALUE" ASC NULLS FIRST"#);
    Ok(())
}

#[test]
fn snowflake_flatten_limit_between_projection_and_unnest_with_subquery_alias()
-> Result<(), DataFusionError> {
    // Build: Projection → Limit → Unnest → SubqueryAlias → Projection → TableScan
    // Combines the Limit and SubqueryAlias transparent node patterns.
    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .alias("t")?
        .unnest_column("__unnest_placeholder(items)")?
        .limit(0, Some(10))?
        .project(vec![col("__unnest_placeholder(items)").alias("item")])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    insta::assert_snapshot!(actual, @r#"SELECT "_unnest_1"."VALUE" AS "item" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1" LIMIT 10"#);
    Ok(())
}

#[test]
fn snowflake_flatten_composed_expression_wrapping_unnest() -> Result<(), DataFusionError>
{
    // Build: Projection(CAST(placeholder AS Int64) AS item_id) → Unnest → Projection → TableScan
    // The outer Projection wraps the unnest output in a function call.
    // The FLATTEN code path must detect the placeholder inside the function
    // and still emit LATERAL FLATTEN.
    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .unnest_column("__unnest_placeholder(items)")?
        .project(vec![
            cast(col("__unnest_placeholder(items)"), DataType::Int64).alias("item_id"),
        ])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    // Must contain LATERAL FLATTEN despite the placeholder being inside CAST
    insta::assert_snapshot!(actual, @r#"SELECT CAST("_unnest_1"."VALUE" AS BIGINT) AS "item_id" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1""#);
    Ok(())
}

#[test]
fn snowflake_flatten_composed_expression_with_limit() -> Result<(), DataFusionError> {
    // Combines both bugs: composed expression + Limit between Projection and Unnest
    // Build: Projection(CAST(placeholder AS Int64) AS item_id) → Limit → Unnest → Projection → TableScan
    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .unnest_column("__unnest_placeholder(items)")?
        .limit(0, Some(5))?
        .project(vec![
            cast(col("__unnest_placeholder(items)"), DataType::Int64).alias("item_id"),
        ])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    insta::assert_snapshot!(actual, @r#"SELECT CAST("_unnest_1"."VALUE" AS BIGINT) AS "item_id" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1" LIMIT 5"#);
    Ok(())
}

#[test]
fn snowflake_flatten_multi_expression_projection() -> Result<(), DataFusionError> {
    // Build: Projection([CAST(placeholder AS Int64) AS a, CAST(placeholder AS Utf8) AS b])
    //          → Unnest → Projection → TableScan
    // The outer Projection has TWO expressions — both reference the placeholder.
    // The FLATTEN code path must fire even when p.expr.len() > 1.
    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .unnest_column("__unnest_placeholder(items)")?
        .project(vec![
            cast(col("__unnest_placeholder(items)"), DataType::Int64).alias("a"),
            cast(col("__unnest_placeholder(items)"), DataType::Utf8).alias("b"),
        ])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    insta::assert_snapshot!(actual, @r#"SELECT CAST("_unnest_1"."VALUE" AS BIGINT) AS "a", CAST("_unnest_1"."VALUE" AS VARCHAR) AS "b" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1""#);
    Ok(())
}

#[test]
fn snowflake_flatten_multi_expression_with_limit() -> Result<(), DataFusionError> {
    // Multi-expression + Limit between Projection and Unnest
    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .unnest_column("__unnest_placeholder(items)")?
        .limit(0, Some(10))?
        .project(vec![
            cast(col("__unnest_placeholder(items)"), DataType::Int64).alias("a"),
            cast(col("__unnest_placeholder(items)"), DataType::Utf8).alias("b"),
        ])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let actual = result.to_string();

    insta::assert_snapshot!(actual, @r#"SELECT CAST("_unnest_1"."VALUE" AS BIGINT) AS "a", CAST("_unnest_1"."VALUE" AS VARCHAR) AS "b" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1" LIMIT 10"#);
    Ok(())
}

#[test]
fn snowflake_unnest_through_subquery_alias() -> Result<(), DataFusionError> {
    // Build: Projection → Unnest → SubqueryAlias → Projection → TableScan
    // This simulates the plan produced when a virtual/passthrough table
    // wraps the source in a SubqueryAlias, which sits between the Unnest
    // and its inner Projection.

    let schema = Schema::new(vec![Field::new(
        "items",
        DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
        true,
    )]);

    let plan = table_scan(Some("source"), &schema, None)?
        .project(vec![col("items").alias("__unnest_placeholder(items)")])?
        .alias("t")? // SubqueryAlias — this is what breaks
        .unnest_column("__unnest_placeholder(items)")?
        .project(vec![col("__unnest_placeholder(items)").alias("item")])?
        .build()?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    let result = unparser.plan_to_sql(&plan)?;
    let sql_str = result.to_string();

    // Should contain LATERAL FLATTEN, not error
    insta::assert_snapshot!(sql_str, @r#"SELECT "_unnest_1"."VALUE" AS "item" FROM "source" CROSS JOIN LATERAL FLATTEN(INPUT => "source"."items", OUTER => true) AS "_unnest_1""#);
    Ok(())
}

#[test]
fn snowflake_flatten_cross_join_unnest_table_column() -> Result<(), DataFusionError> {
    // Single CROSS JOIN UNNEST from a table column with user-provided alias.
    // Column references into the FLATTEN alias use .VALUE.
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT * FROM multi_array_table CROSS JOIN UNNEST(column_a) AS a (a)",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "multi_array_table"."column_a", "multi_array_table"."column_b", "a"."VALUE" FROM "multi_array_table" CROSS JOIN LATERAL FLATTEN(INPUT => "multi_array_table"."column_a") AS "a""#,
    );
    Ok(())
}

#[test]
fn snowflake_flatten_multiple_unnest_cross_join() -> Result<(), DataFusionError> {
    // Realistic Snowflake pattern:
    //   SELECT a, b
    //   FROM multi_array_table
    //   CROSS JOIN UNNEST(column_a) AS a
    //   CROSS JOIN UNNEST(column_b) AS b
    //
    // Each CROSS JOIN UNNEST should produce a separate LATERAL FLATTEN
    // with a distinct alias so they don't collide in the same FROM clause.
    let snowflake = SnowflakeDialect::new();
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT a.a, b.b FROM multi_array_table CROSS JOIN UNNEST(column_a) AS a (a) CROSS JOIN UNNEST(column_b) AS b (b)",
        parser_dialect: GenericDialect {},
        unparser_dialect: snowflake,
        expected: @r#"SELECT "a"."VALUE", "b"."VALUE" FROM "multi_array_table" CROSS JOIN LATERAL FLATTEN(INPUT => "multi_array_table"."column_a") AS "a" CROSS JOIN LATERAL FLATTEN(INPUT => "multi_array_table"."column_b") AS "b""#,
    );
    Ok(())
}
/// Regression test for chained `INTERSECT`/`EXCEPT` (e.g. TPC-DS q38).
///
/// When both intersect/except branches share identical column qualifiers, the
/// planner requalifies the inputs as `left`/`right` subquery aliases and lowers
/// the set operation to a `LeftSemi`/`LeftAnti` join. For a 3-way chain the
/// build (right) side of the *outer* join is a complex plan
/// (`Distinct(Projection(Join))`). Previously the unparser merged that plan's
/// projection, joins, DISTINCT and WHERE into the shared outer SELECT while the
/// generated `EXISTS` captured only the base relation. That produced a spurious
/// join on the outer FROM and correlated references to tables not in scope
/// (Postgres: `missing FROM-clause entry for table "left"`).
///
/// The build side must now be emitted as a self-contained subquery: each
/// `EXISTS` carries its own `FROM ... JOIN ...` so every column reference
/// resolves.
#[test]
fn test_unparse_chained_intersect_build_side_is_self_contained() -> Result<()> {
    let branch = "SELECT DISTINCT p.first_name, o.order_id \
                  FROM person p JOIN orders o ON p.id = o.customer_id";
    let query = format!("{branch} INTERSECT {branch} INTERSECT {branch}");

    let statement = Parser::new(&GenericDialect {})
        .try_with_sql(&query)?
        .parse_statement()?;
    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let plan = SqlToRel::new(&context).sql_statement_to_plan(statement)?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    let sql = unparser.plan_to_sql(&plan)?;
    assert_snapshot!(
        sql,
        @r#"SELECT DISTINCT * FROM (SELECT DISTINCT "p"."first_name", "o"."order_id" FROM "person" AS "p" INNER JOIN "orders" AS "o" ON ("p"."id" = "o"."customer_id")) AS "left" WHERE EXISTS (SELECT 1 FROM (SELECT DISTINCT "p"."first_name", "o"."order_id" FROM "person" AS "p" INNER JOIN "orders" AS "o" ON ("p"."id" = "o"."customer_id")) AS "right" WHERE ("left"."first_name" = "right"."first_name") AND ("left"."order_id" = "right"."order_id")) AND EXISTS (SELECT 1 FROM "person" AS "p" INNER JOIN "orders" AS "o" ON ("p"."id" = "o"."customer_id") WHERE ("left"."first_name" = "p"."first_name") AND ("left"."order_id" = "o"."order_id"))"#
    );
    Ok(())
}

/// A `Filter` above a row limit must not be flattened into the same `SELECT`
/// as that limit: SQL evaluates `WHERE` before `LIMIT`, so the flattened form
/// means the opposite of the plan and can return rows the plan excludes.
#[test]
fn test_filter_above_limit_gets_its_own_scope() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    // Take 5 rows, then keep the matching ones. The limit belongs in a derived
    // table, named after the relation the surviving predicate is qualified by.
    let plan = table_scan(Some("t"), &schema, None)?
        .limit(0, Some(5))?
        .filter(col("id").eq(lit("a")))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT t.id, t."name" FROM (SELECT * FROM t LIMIT 5) AS t WHERE (t.id = 'a')"#
    );

    // An OFFSET is evaluated at the same point as a LIMIT, so a skip-only
    // limit reorders against the predicate in exactly the same way.
    let plan = table_scan(Some("t"), &schema, None)?
        .limit(3, None)?
        .filter(col("id").eq(lit("a")))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT t.id, t."name" FROM (SELECT * FROM t OFFSET 3) AS t WHERE (t.id = 'a')"#
    );

    Ok(())
}

/// Builds `count(<col>)` with the aggregate-function stub used across these tests.
fn count_col(name: &str) -> Expr {
    use datafusion_expr::expr::{AggregateFunction, AggregateFunctionParams};
    Expr::AggregateFunction(AggregateFunction {
        func: count_udaf(),
        params: AggregateFunctionParams {
            args: vec![col(name)],
            distinct: false,
            filter: None,
            order_by: vec![],
            null_treatment: None,
        },
    })
}

/// A SELECT carries a single grouping, so an aggregate stacked directly on top of
/// another has to be unparsed as a derived table. This is the plan
/// `single_distinct_to_groupby` produces for `count(DISTINCT b)`: an outer
/// `count(alias1)` over an inner `GROUP BY b AS alias1`. Folding both into one SELECT
/// emits `count(alias1)` against the base table, where `alias1` does not exist — and
/// where a column of that name happens to exist, the DISTINCT is silently dropped.
#[test]
fn stacked_aggregate_is_unparsed_as_a_derived_table() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
    ]);

    // count(DISTINCT b) — the outer aggregate groups by nothing.
    let plan = table_scan(Some("test"), &schema, None)?
        .aggregate(vec![col("test.b").alias("alias1")], Vec::<Expr>::new())?
        .aggregate(Vec::<Expr>::new(), vec![count_col("alias1")])?
        .project(vec![col("COUNT(alias1)").alias("count(DISTINCT test.b)")])?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT COUNT(alias1) AS "count(DISTINCT test.b)" FROM (SELECT test.b AS alias1 FROM test GROUP BY test.b) AS derived_aggregate_1"#
    );

    // a, count(DISTINCT b) ... GROUP BY a — the outer aggregate keeps its own grouping,
    // which must not absorb the inner one.
    let plan = table_scan(Some("test"), &schema, None)?
        .aggregate(
            vec![col("test.a"), col("test.b").alias("alias1")],
            Vec::<Expr>::new(),
        )?
        .aggregate(vec![col("test.a")], vec![count_col("alias1")])?
        .project(vec![
            col("test.a"),
            col("COUNT(alias1)").alias("count(DISTINCT test.b)"),
        ])?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT derived_aggregate_1.a, COUNT(alias1) AS "count(DISTINCT test.b)" FROM (SELECT test.a, test.b AS alias1 FROM test GROUP BY test.a, test.b) AS derived_aggregate_1 GROUP BY derived_aggregate_1.a"#
    );

    // A lone aggregate is still folded into the SELECT it belongs to.
    let plan = table_scan(Some("test"), &schema, None)?
        .aggregate(vec![col("test.a")], vec![count_col("test.b")])?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @"SELECT COUNT(test.b), test.a FROM test GROUP BY test.a"
    );

    Ok(())
}

/// Once the inner aggregate becomes a derived table, the enclosing SELECT reads from that
/// derived table and not from `test`, so a reference still qualified by `test` binds to
/// nothing. DataFusion re-plans such a query, but PostgreSQL answers 42703 and DuckDB a
/// Binder Error, so the failure surfaces as a driver error on a federated pushdown.
#[test]
fn stacked_aggregate_requalifies_every_clause_onto_the_derived_table() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
    ]);
    let stacked = || -> Result<LogicalPlanBuilder> {
        table_scan(Some("test"), &schema, None)?.aggregate(
            vec![col("test.a"), col("test.b").alias("alias1")],
            Vec::<Expr>::new(),
        )
    };

    // GROUP BY on a bare qualified column.
    let plan = stacked()?
        .aggregate(vec![col("test.a")], vec![count_col("alias1")])?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @"SELECT COUNT(alias1), derived_aggregate_1.a FROM (SELECT test.a, test.b AS alias1 FROM test GROUP BY test.a, test.b) AS derived_aggregate_1 GROUP BY derived_aggregate_1.a"
    );

    // A qualifier nested inside a grouping expression is reached too — the projection and
    // the GROUP BY have to agree, or the GROUP BY no longer covers the selected expression.
    let plan = stacked()?
        .aggregate(vec![col("test.a") + lit(1u32)], vec![count_col("alias1")])?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @"SELECT COUNT(alias1), (derived_aggregate_1.a + 1) FROM (SELECT test.a, test.b AS alias1 FROM test GROUP BY test.a, test.b) AS derived_aggregate_1 GROUP BY (derived_aggregate_1.a + 1)"
    );

    // HAVING is built from a Filter above the aggregate and was never swept.
    let plan = stacked()?
        .aggregate(vec![col("test.a")], vec![count_col("alias1")])?
        .filter(col("test.a").gt(lit(1u32)))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @"SELECT COUNT(alias1), derived_aggregate_1.a FROM (SELECT test.a, test.b AS alias1 FROM test GROUP BY test.a, test.b) AS derived_aggregate_1 GROUP BY derived_aggregate_1.a HAVING (derived_aggregate_1.a > 1)"
    );

    // A query-level ORDER BY is reached by the existing dangling-identifier sweep, which
    // runs later and strips the qualifier rather than repointing it. It binds either way;
    // this pins that the two mechanisms do not fight over the same clause.
    let plan = stacked()?
        .aggregate(vec![col("test.a")], vec![count_col("alias1")])?
        .sort(vec![col("test.a").sort(true, false)])?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @"SELECT COUNT(alias1), derived_aggregate_1.a FROM (SELECT test.a, test.b AS alias1 FROM test GROUP BY test.a, test.b) AS derived_aggregate_1 GROUP BY derived_aggregate_1.a ORDER BY a ASC NULLS LAST"
    );

    Ok(())
}

/// The derived table is aliased for every dialect, so a dialect that also *requires* the
/// alias must not end up with a second one, and its references resolve the same way.
#[test]
fn stacked_aggregate_requalifies_onto_a_required_derived_table_alias() -> Result<()> {
    struct AliasedDerivedTableDialect {}
    impl UnparserDialect for AliasedDerivedTableDialect {
        fn identifier_quote_style(&self, _: &str) -> Option<char> {
            None
        }
        fn requires_derived_table_alias(&self) -> bool {
            true
        }
    }

    let schema = Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
    ]);
    let plan = table_scan(Some("test"), &schema, None)?
        .aggregate(
            vec![col("test.a"), col("test.b").alias("alias1")],
            Vec::<Expr>::new(),
        )?
        .aggregate(vec![col("test.a")], vec![count_col("alias1")])?
        .filter(col("test.a").gt(lit(1u32)))?
        .build()?;

    let unparser = Unparser::new(&AliasedDerivedTableDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @"SELECT COUNT(alias1), derived_aggregate_1.a FROM (SELECT test.a, test.b AS alias1 FROM test GROUP BY test.a, test.b) AS derived_aggregate_1 GROUP BY derived_aggregate_1.a HAVING (derived_aggregate_1.a > 1)"
    );

    Ok(())
}

/// A join shares one `SelectBuilder` across both sides, and the join is recorded on it only
/// after the left side has been walked — so at the point the derived table is built, this
/// SELECT reads from more than one relation even though nothing on the builder says so yet.
/// A reference qualified by the *other* side must keep its qualifier, and one pointed at the
/// derived table must name it rather than go bare, since `a` here is ambiguous between the
/// two relations.
#[test]
fn stacked_aggregate_under_a_join_leaves_the_other_side_qualified() -> Result<()> {
    let left = Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
    ]);
    // `other` deliberately repeats the column name `a`.
    let right = Schema::new(vec![Field::new("a", DataType::UInt32, false)]);

    let inner_aggregate = table_scan(Some("test"), &left, None)?
        .aggregate(vec![col("test.a")], Vec::<Expr>::new())?
        .build()?;

    let plan = LogicalPlanBuilder::from(inner_aggregate)
        .join_on(
            table_scan(Some("other"), &right, None)?.build()?,
            datafusion_expr::JoinType::Inner,
            vec![col("test.a").eq(col("other.a"))],
        )?
        .aggregate(vec![col("other.a")], vec![count_col("test.a")])?
        .build()?;

    // `COUNT(derived_aggregate_1.a)` and `GROUP BY "other".a` stay distinguishable. Reducing
    // either to a bare `a` would make the query ambiguous, and group by the wrong column.
    //
    // The join's `ON` still reads `test.a`: it is built as part of the join and attached
    // after this SELECT's clauses, so it is not among the clauses that are swept. That is
    // unchanged by the requalification and is tracked as spiceai/spiceai#12695; pinning it
    // here keeps the gap visible rather than silent.
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT COUNT(derived_aggregate_1.a), "other".a FROM (SELECT test.a FROM test GROUP BY test.a) AS derived_aggregate_1 INNER JOIN "other" ON (test.a = "other".a) GROUP BY "other".a"#
    );

    Ok(())
}

/// Both sides of a join can carry a stacked aggregate, and one `SelectBuilder` walks both,
/// so both derive a table into the same FROM clause. Their aliases have to differ on two
/// counts: a repeated name is a duplicate table name most engines reject outright, and —
/// the quieter half — each side requalifies its own references onto its own alias, so a
/// shared name would collapse the two sides' distinct columns onto one qualifier and group
/// by whichever side was walked first.
#[test]
fn stacked_aggregates_on_both_join_sides_get_distinct_aliases() -> Result<()> {
    let left = Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
    ]);
    // `other` deliberately repeats the column name `a`, so a collapsed qualifier would
    // still bind — and silently read the wrong side — rather than fail loudly.
    let right = Schema::new(vec![Field::new("a", DataType::UInt32, false)]);

    let left_aggregate = table_scan(Some("test"), &left, None)?
        .aggregate(vec![col("test.a")], Vec::<Expr>::new())?
        .build()?;
    let right_aggregate = table_scan(Some("other"), &right, None)?
        .aggregate(vec![col("other.a")], Vec::<Expr>::new())?
        .build()?;

    let plan = LogicalPlanBuilder::from(left_aggregate)
        .join_on(
            right_aggregate,
            datafusion_expr::JoinType::Inner,
            vec![col("test.a").eq(col("other.a"))],
        )?
        .aggregate(vec![col("other.a")], vec![count_col("test.a")])?
        .build()?;

    // `COUNT` keeps the left side and `GROUP BY` the right, each through its own alias.
    // The join's `ON` still reads the pre-derivation qualifiers, which is the gap tracked
    // as spiceai/spiceai#12695 and is pinned here rather than left silent.
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT COUNT(derived_aggregate_1.a), derived_aggregate_2.a FROM (SELECT test.a FROM test GROUP BY test.a) AS derived_aggregate_1 INNER JOIN (SELECT "other".a FROM "other" GROUP BY "other".a) AS derived_aggregate_2 ON (test.a = "other".a) GROUP BY derived_aggregate_2.a"#
    );

    Ok(())
}

/// A correlated subquery references an enclosing query's relation by a qualifier that is
/// indistinguishable, by name, from one addressing this SELECT's own relation — here both
/// are `test.a`. Rewriting inside the subquery would repoint it at the derived table and
/// silently change which column it reads, so an expression holding a subquery is left
/// alone entirely.
///
/// The cost is visible below: the `test.a` on the left of the comparison is this SELECT's
/// own and would otherwise be requalified, but it shares an expression with the subquery
/// and so keeps its qualifier. That is the pre-existing output for this shape, not a
/// regression, and it is the conservative direction — an unbindable qualifier is a loud
/// error at the remote engine, whereas a repointed correlated reference returns wrong rows.
#[test]
fn stacked_aggregate_leaves_a_correlated_outer_reference_alone() -> Result<()> {
    let outer = Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
    ]);
    let inner = Schema::new(vec![Field::new("c", DataType::UInt32, false)]);

    // HAVING count(alias1) > (SELECT count(other.c) FROM other WHERE other.c = test.a)
    let correlated = scalar_subquery(Arc::new(
        table_scan(Some("other"), &inner, None)?
            .filter(col("other.c").eq(out_ref_col(DataType::UInt32, "test.a")))?
            .aggregate(Vec::<Expr>::new(), vec![count_col("other.c")])?
            .build()?,
    ));

    let plan = table_scan(Some("test"), &outer, None)?
        .aggregate(
            vec![col("test.a"), col("test.b").alias("alias1")],
            Vec::<Expr>::new(),
        )?
        .aggregate(vec![col("test.a")], vec![count_col("alias1")])?
        .filter(col("test.a").gt(correlated))?
        .build()?;

    // `other.c = test.a` inside the subquery still reads the outer `test.a`, which is the
    // property that matters. `GROUP BY a` shows the requalification still runs on the
    // clauses that hold no subquery.
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT COUNT(alias1), derived_aggregate_1.a FROM (SELECT test.a, test.b AS alias1 FROM test GROUP BY test.a, test.b) AS derived_aggregate_1 GROUP BY derived_aggregate_1.a HAVING (test.a > (SELECT COUNT("other".c) FROM "other" WHERE ("other".c = test.a)))"#
    );

    Ok(())
}

/// A `Sort` that has to be emitted below the SELECT list must not end up inside a
/// derived table: SQL does not require an enclosing query to honour the ORDER BY of
/// a derived table, so the rows come back in an arbitrary order.
#[test]
fn order_by_over_non_projected_field_stays_top_level() -> Result<()> {
    // Sort key is an expression over a column the SELECT list does not project.
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM person ORDER BY age + 1 DESC",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT person.id FROM person ORDER BY (person.age + 1) DESC NULLS FIRST",
    );

    // Same, with the sort key an expression over an aggregate that is not selected.
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id, first_name FROM person GROUP BY id, first_name ORDER BY max(age) + 1 DESC",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT person.id, person.first_name FROM person GROUP BY person.id, person.first_name ORDER BY (max(person.age) + 1) DESC NULLS FIRST",
    );

    // A plain non-projected column already worked; keep it covered so the
    // generalisation above cannot regress it.
    roundtrip_statement_with_dialect_helper!(
        sql: "SELECT id FROM person ORDER BY age DESC",
        parser_dialect: GenericDialect {},
        unparser_dialect: UnparserDefaultDialect {},
        expected: @"SELECT person.id FROM person ORDER BY person.age DESC NULLS FIRST",
    );

    Ok(())
}

/// A `Sort` carrying a `fetch` renders that fetch as the query's `LIMIT`, so a
/// predicate above it reorders in the same way. It cannot take the same fix: a
/// derived table's row order is not carried out to the query selecting from
/// it, so moving the sort inside one would repair the row set and lose the
/// ordering the plan promises.
#[test]
fn test_filter_above_a_sort_fetch_is_refused() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let plan = table_scan(Some("t"), &schema, None)?
        .sort_with_limit(vec![col("id").sort(true, false)], Some(5))?
        .filter(col("id").eq(lit("a")))?
        .build()?;
    let error =
        plan_to_sql(&plan).expect_err("a filter above a sort fetch cannot be unparsed");
    assert_contains!(error.to_string(), "after a sort's fetch");

    Ok(())
}

/// The two predicate sources around a limit are applied at different times: a
/// `TableScan`'s own filters run before the scan's rows reach the limit, a
/// `Filter` node above the limit runs after. Both are accumulated into the
/// same `WHERE` while the walk is inside one `SELECT`, so the split has to
/// come from where each one is unparsed, not from the clause it lands in.
#[test]
fn test_scan_filter_stays_below_the_limit_it_precedes() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let plan = table_scan_with_filters(
        Some("t"),
        &schema,
        None,
        vec![col("name").eq(lit("z"))],
    )?
    .limit(0, Some(5))?
    .filter(col("id").eq(lit("a")))?
    .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT t.id, t."name" FROM (SELECT * FROM t WHERE (t."name" = 'z') LIMIT 5) AS t WHERE (t.id = 'a')"#
    );

    Ok(())
}

/// The new scope is only for a predicate that sits *above* the limit. A
/// predicate below it is already in the order SQL evaluates, and a limit with
/// no predicate above it has nothing to reorder against — neither may grow a
/// derived table.
#[test]
fn test_limit_without_a_predicate_above_it_stays_flat() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    // Filter below the limit: keep the matching rows, then take 5 of them —
    // which is what a single `SELECT ... WHERE ... LIMIT` already means.
    let plan = table_scan(Some("t"), &schema, None)?
        .filter(col("id").eq(lit("a")))?
        .limit(0, Some(5))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT * FROM t WHERE (t.id = 'a') LIMIT 5"#
    );

    // No predicate at all.
    let plan = table_scan(Some("t"), &schema, None)?
        .limit(0, Some(5))?
        .build()?;
    assert_snapshot!(plan_to_sql(&plan)?, @r#"SELECT * FROM t LIMIT 5"#);

    // A sort without a fetch contributes no limit, so a predicate above it is
    // still evaluated first either way.
    let plan = table_scan(Some("t"), &schema, None)?
        .sort(vec![col("id").sort(true, false)])?
        .filter(col("id").eq(lit("a")))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT * FROM t WHERE (t.id = 'a') ORDER BY t.id ASC NULLS LAST"#
    );

    Ok(())
}

/// `HAVING` is evaluated before `LIMIT` too, so an aggregate-referencing
/// filter above a limit reorders in the same way a `WHERE` does — but it
/// cannot be lifted out the way a `WHERE` can, because the aggregate it
/// references is only nameable in the `SELECT` that computes it. Refusing is
/// the only answer that is neither reversed nor unbindable.
#[test]
fn test_having_above_limit_is_refused() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let plan = table_scan(Some("t"), &schema, None)?
        .aggregate(vec![col("name")], vec![count(col("id"))])?
        .limit(0, Some(5))?
        .filter(col("COUNT(t.id)").gt(lit(1i64)))?
        .build()?;
    let error =
        plan_to_sql(&plan).expect_err("a HAVING above a limit cannot be unparsed");
    assert_contains!(
        error.to_string(),
        "HAVING or QUALIFY predicate that is applied after a row limit"
    );

    // The same aggregate with the limit above the filter is the order SQL
    // already means, and still unparses.
    let plan = table_scan(Some("t"), &schema, None)?
        .aggregate(vec![col("name")], vec![count(col("id"))])?
        .filter(col("COUNT(t.id)").gt(lit(1i64)))?
        .limit(0, Some(5))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT COUNT(t.id), t."name" FROM t GROUP BY t."name" HAVING (COUNT(t.id) > 1) LIMIT 5"#
    );

    Ok(())
}

/// The generated SQL has to mean what the plan meant, and the derived table
/// has to be addressable by the predicate left outside it. Planning the SQL
/// back proves both: an unresolvable qualifier would fail to plan, and a
/// reordered one would come back with the limit above the filter.
#[test]
fn test_filter_above_limit_round_trips() -> Result<()> {
    let schema = Schema::new(vec![Field::new("id", DataType::UInt32, false)]);
    let plan = table_scan(Some("person"), &schema, None)?
        .limit(0, Some(5))?
        .filter(col("id").gt(lit(5u32)))?
        .build()?;

    let sql = plan_to_sql(&plan)?;
    let statement = Parser::new(&GenericDialect {})
        .try_with_sql(&sql.to_string())?
        .parse_statement()?;
    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let replanned = SqlToRel::new(&context).sql_statement_to_plan(statement)?;

    let displayed = replanned.display_indent().to_string();
    let filter_at = displayed
        .find("Filter:")
        .expect("re-planned SQL should still filter");
    let limit_at = displayed
        .find("Limit:")
        .expect("re-planned SQL should still limit");
    assert!(
        filter_at < limit_at,
        "the limit must stay below the filter, got:\n{displayed}"
    );

    Ok(())
}

/// A limited subtree that reads two relations cannot become a derived table
/// the outer predicate can still address: both qualifiers leave scope, and no
/// single name replaces them. Refuse, rather than reverse the two or emit a
/// predicate with nothing to bind to.
#[test]
fn test_filter_above_limit_over_a_join_is_refused() -> Result<()> {
    let left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let plan = LogicalPlanBuilder::from(table_scan(Some("a"), &left, None)?.build()?)
        .join(
            table_scan(Some("b"), &right, None)?.build()?,
            datafusion_expr::JoinType::Inner,
            (vec!["a.id"], vec!["b.id"]),
            None,
        )?
        .limit(0, Some(5))?
        .filter(col("a.name").eq(lit("x")))?
        .build()?;
    let error = plan_to_sql(&plan)
        .expect_err("a filter above a limit over a join cannot be unparsed");
    assert_contains!(error.to_string(), "limited input is a single table scan");

    Ok(())
}

/// A `SubqueryAlias` under the limit already supplies the name the outer
/// predicate is qualified by, so it is the name the derived table takes.
#[test]
fn test_filter_above_limit_keeps_a_subquery_alias() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let plan = table_scan(Some("t"), &schema, None)?
        .alias("x")?
        .limit(0, Some(5))?
        .filter(col("x.name").eq(lit("q")))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT x.id, x."name" FROM (SELECT * FROM t AS x LIMIT 5) AS x WHERE (x."name" = 'q')"#
    );

    Ok(())
}

/// A limit on one join input is that input's own, and the derived table it
/// becomes contributes only that input's columns to the enclosing `SELECT`
/// list. A wildcard there would expand to every relation in the `FROM`,
/// returning the other side's columns twice.
#[test]
fn test_filter_above_a_limited_join_input_keeps_the_join_schema() -> Result<()> {
    let left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let limited_left = table_scan(Some("a"), &left, Some(vec![0, 1]))?
        .limit(0, Some(5))?
        .build()?;
    let plan = LogicalPlanBuilder::from(limited_left)
        .join(
            table_scan(Some("b"), &right, Some(vec![0, 1]))?.build()?,
            datafusion_expr::JoinType::Inner,
            (vec!["a.id"], vec!["b.id"]),
            None,
        )?
        .filter(col("a.name").eq(lit("x")))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT a.id, a."name", b.id, b.age FROM (SELECT a.id, a."name" FROM a LIMIT 5) AS a INNER JOIN b ON a.id = b.id WHERE (a."name" = 'x')"#
    );

    Ok(())
}

/// The derived table's columns are named by the enclosing query, which only
/// works if the derived query names them the same way. A projection breaks
/// that: an unaliased expression is a column the derived query never names,
/// and two aliases differing only by a qualifier collapse onto one SQL name.
#[test]
fn test_filter_above_limit_over_a_projection_is_refused() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);

    let plan = table_scan(Some("t"), &schema, None)?
        .project(vec![col("a").add(col("b"))])?
        .limit(0, Some(5))?
        .filter(col("t.a + t.b").gt(lit(1i32)))?
        .build()?;
    let error = plan_to_sql(&plan)
        .expect_err("a filter above a limited projection cannot be unparsed");
    assert_contains!(error.to_string(), "limited input is a single table scan");

    Ok(())
}

/// A sort below the limit orders what the plan returns just as surely as one
/// above it, and a derived table does not carry its row order out. It is the
/// one clause that may wrap a scan and still not be safe to move inside.
#[test]
fn test_filter_above_a_limited_sort_is_refused() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let plan = table_scan(Some("t"), &schema, None)?
        .sort(vec![col("id").sort(true, false)])?
        .limit(0, Some(5))?
        .filter(col("id").eq(lit("a")))?
        .build()?;
    let error =
        plan_to_sql(&plan).expect_err("a filter above a limited sort cannot be unparsed");
    assert_contains!(error.to_string(), "limited input is a single table scan");

    Ok(())
}

/// A dialect that spells columns in full leaves the outer predicate qualified
/// by every part of the table's path, while a derived table can only be
/// aliased by one identifier. The predicate would be left naming a path that
/// is no longer in scope.
#[test]
fn test_filter_above_limit_is_refused_for_a_fully_qualified_column() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .with_identifier_quote_style('"')
        .build();

    let plan = table_scan(Some("catalog.schema.t"), &schema, None)?
        .limit(0, Some(5))?
        .filter(col("id").eq(lit("a")))?
        .build()?;
    let error = Unparser::new(&dialect)
        .plan_to_sql(&plan)
        .expect_err("a fully qualified predicate cannot survive the alias");
    assert_contains!(error.to_string(), "dialect that spells columns in full");

    // A bare table name has nothing to lose, so the same dialect renders it.
    let plan = table_scan(Some("t"), &schema, None)?
        .limit(0, Some(5))?
        .filter(col("id").eq(lit("a")))?
        .build()?;
    assert_snapshot!(
        Unparser::new(&dialect).plan_to_sql(&plan)?,
        @r#"SELECT "t"."id", "t"."name" FROM (SELECT * FROM "t" LIMIT 5) AS "t" WHERE ("t"."id" = 'a')"#
    );

    Ok(())
}

/// Join inputs are walked with one shared `SelectBuilder`, so a predicate on
/// it need not be an ancestor of the node being unparsed. A `HAVING` from one
/// input must not make the other input's limit unrenderable.
#[test]
fn test_a_sibling_having_does_not_refuse_the_other_inputs_limit() -> Result<()> {
    let left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let grouped_left = table_scan(Some("a"), &left, None)?
        .aggregate(vec![col("id")], vec![count(col("name"))])?
        .filter(col("COUNT(a.name)").gt(lit(1i64)))?
        .build()?;
    let limited_right = table_scan(Some("b"), &right, Some(vec![0, 1]))?
        .limit(0, Some(5))?
        .build()?;
    let plan = LogicalPlanBuilder::from(grouped_left)
        .join(
            limited_right,
            datafusion_expr::JoinType::Inner,
            (vec!["a.id"], vec!["b.id"]),
            None,
        )?
        .build()?;

    plan_to_sql(&plan).expect("a sibling's HAVING is not this limit's predicate");

    Ok(())
}

/// A scan can project no columns, and then there is no column list to name
/// the derived table's output with.
#[test]
fn test_filter_above_limit_over_an_empty_projection_is_refused() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let plan = table_scan(Some("t"), &schema, Some(vec![]))?
        .limit(0, Some(5))?
        .filter(lit(true))?
        .build()?;
    let error = plan_to_sql(&plan)
        .expect_err("a limited scan with no columns cannot be unparsed");
    assert_contains!(error.to_string(), "projecting no columns");

    Ok(())
}

#[test]
fn test_join_filter_nested_under_null_extending_join() -> Result<()> {
    // When an enclosing RIGHT JOIN null-extends a nested left input, a predicate
    // from that input must not reach the SELECT-global `WHERE`: `WHERE` runs
    // after every join and would discard the preserved right-side rows. It
    // belongs in the enclosing join's `ON` because the left input is not
    // preserved. FULL JOIN differs because it preserves both inputs.
    let schema = Schema::new(vec![Field::new("id", DataType::Utf8, false)]);
    let a = table_scan_with_filters(
        Some("a"),
        &schema,
        Some(vec![0]),
        vec![col("a.id").eq(lit("x"))],
    )?
    .build()?;
    let b = table_scan(Some("b"), &schema, Some(vec![0]))?.build()?;
    let c = table_scan(Some("c"), &schema, Some(vec![0]))?.build()?;

    let nested_under_outer = |inner_join_type, outer_join_type| -> Result<String> {
        let inner = LogicalPlanBuilder::from(a.clone())
            .join(
                b.clone(),
                inner_join_type,
                (vec!["a.id"], vec!["b.id"]),
                None,
            )?
            .build()?;
        let outer = LogicalPlanBuilder::from(inner)
            .join(
                c.clone(),
                outer_join_type,
                (vec!["a.id"], vec!["c.id"]),
                None,
            )?
            .build()?;
        Ok(plan_to_sql(&outer)?.to_string())
    };

    assert_snapshot!(
        nested_under_outer(
            datafusion_expr::JoinType::Inner,
            datafusion_expr::JoinType::Right,
        )?,
        @"SELECT a.id, b.id, c.id FROM a INNER JOIN b ON a.id = b.id RIGHT OUTER JOIN c ON a.id = c.id AND (a.id = 'x')"
    );
    assert_snapshot!(
        nested_under_outer(
            datafusion_expr::JoinType::Left,
            datafusion_expr::JoinType::Right,
        )?,
        @"SELECT a.id, b.id, c.id FROM a LEFT OUTER JOIN b ON a.id = b.id RIGHT OUTER JOIN c ON a.id = c.id AND (a.id = 'x')"
    );

    // A FULL JOIN also null-extends its left input, but unlike a RIGHT JOIN it
    // preserves that input. Hoisting the predicate into ON would therefore
    // make filtered-out left rows reappear as unmatched rows. Keep the prior
    // WHERE placement until FULL JOIN inputs can be emitted as derived tables.
    assert_snapshot!(
        nested_under_outer(
            datafusion_expr::JoinType::Inner,
            datafusion_expr::JoinType::Full,
        )?,
        @"SELECT a.id, b.id, c.id FROM a INNER JOIN b ON a.id = b.id FULL JOIN c ON a.id = c.id WHERE (a.id = 'x')"
    );

    Ok(())
}

/// The inner `Projection` may define an alias that only the `Sort` uses. Hoisting the
/// `Sort` drops that alias, so every reference to it -- including one nested inside a
/// larger sort expression -- has to be replaced by the expression it named.
#[test]
fn order_by_nested_reference_to_dropped_alias() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("age", DataType::Int64, false),
    ]);
    let plan = table_scan(Some("person"), &schema, None)?
        .project(vec![col("id"), (col("age") * lit(2)).alias("doubled")])?
        .sort(vec![(col("doubled") + lit(1)).sort(false, true)])?
        .project(vec![col("id")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @"SELECT person.id FROM person ORDER BY ((person.age * 2) + 1) DESC NULLS FIRST"
    );

    Ok(())
}

#[test]
fn test_join_filter_nested_under_right_join_using() -> Result<()> {
    // USING cannot carry the predicate contributed by the non-preserved left
    // input. It must be downgraded to an equivalent ON constraint before the
    // predicate is appended; returning the predicate to the SELECT-global
    // WHERE would discard unmatched rows from the preserved right input.
    let id_schema = Schema::new(vec![Field::new("id", DataType::Utf8, false)]);
    let b_schema = Schema::new(vec![Field::new("b_id", DataType::Utf8, false)]);
    let a = table_scan_with_filters(
        Some("a"),
        &id_schema,
        Some(vec![0]),
        vec![col("a.id").eq(lit("x"))],
    )?
    .build()?;
    let b = table_scan(Some("b"), &b_schema, Some(vec![0]))?.build()?;
    let c = table_scan(Some("c"), &id_schema, Some(vec![0]))?.build()?;

    let inner = LogicalPlanBuilder::from(a)
        .join(
            b,
            datafusion_expr::JoinType::Inner,
            (vec!["a.id"], vec!["b.b_id"]),
            None,
        )?
        .build()?;
    let outer = LogicalPlanBuilder::from(inner)
        .join_using(
            c,
            datafusion_expr::JoinType::Right,
            vec![Column::new_unqualified("id")],
        )?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&outer)?,
        @"SELECT a.id, b.b_id, c.id FROM a INNER JOIN b ON a.id = b.b_id RIGHT OUTER JOIN c ON a.id = c.id AND (a.id = 'x')"
    );

    Ok(())
}

/// A `fetch` pushed into a join input bounds that input, not the join's output,
/// so it has to be emitted as a derived table. Dropping it asks the remote
/// engine for the whole table and joins over it.
#[test]
fn test_join_input_with_pushed_down_fetch() -> Result<()> {
    let schema_left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let schema_right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let scan_with_fetch = |name: &'static str, schema: &Schema, filters: Vec<Expr>| {
        table_scan_with_filter_and_fetch(
            Some(name),
            schema,
            Some(vec![0, 1]),
            filters,
            Some(5),
        )
    };

    // Left input: filter and fetch both move into the subquery. Keeping the
    // filter outside would apply it after the limit, which returns fewer rows.
    let plan = LogicalPlanBuilder::from(
        scan_with_fetch(
            "left_table",
            &schema_left,
            vec![col("left_table.id").eq(lit("a"))],
        )?
        .build()?,
    )
    .join(
        table_scan(Some("right_table"), &schema_right, Some(vec![0, 1]))?.build()?,
        datafusion_expr::JoinType::Inner,
        (vec!["left_table.id"], vec!["right_table.id"]),
        None,
    )?
    .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM (SELECT left_table.id, left_table."name" FROM left_table WHERE (left_table.id = 'a') LIMIT 5) AS left_table INNER JOIN right_table ON left_table.id = right_table.id"#
    );

    // Right input: the same, on the other side of the join.
    let plan = LogicalPlanBuilder::from(
        table_scan(Some("left_table"), &schema_left, Some(vec![0, 1]))?.build()?,
    )
    .join(
        scan_with_fetch(
            "right_table",
            &schema_right,
            vec![col("right_table.age").gt(lit(30))],
        )?
        .build()?,
        datafusion_expr::JoinType::Inner,
        (vec!["left_table.id"], vec!["right_table.id"]),
        None,
    )?
    .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table INNER JOIN (SELECT right_table.id, right_table.age FROM right_table WHERE (right_table.age > 30) LIMIT 5) AS right_table ON left_table.id = right_table.id"#
    );

    // A fetch with no filter still needs the subquery.
    let plan = LogicalPlanBuilder::from(
        scan_with_fetch("left_table", &schema_left, vec![])?.build()?,
    )
    .join(
        table_scan(Some("right_table"), &schema_right, Some(vec![0, 1]))?.build()?,
        datafusion_expr::JoinType::Inner,
        (vec!["left_table.id"], vec!["right_table.id"]),
        None,
    )?
    .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM (SELECT left_table.id, left_table."name" FROM left_table LIMIT 5) AS left_table INNER JOIN right_table ON left_table.id = right_table.id"#
    );

    // Both inputs limited: each gets its own subquery.
    let plan = LogicalPlanBuilder::from(
        scan_with_fetch("left_table", &schema_left, vec![])?.build()?,
    )
    .join(
        scan_with_fetch("right_table", &schema_right, vec![])?.build()?,
        datafusion_expr::JoinType::Inner,
        (vec!["left_table.id"], vec!["right_table.id"]),
        None,
    )?
    .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM (SELECT left_table.id, left_table."name" FROM left_table LIMIT 5) AS left_table INNER JOIN (SELECT right_table.id, right_table.age FROM right_table LIMIT 5) AS right_table ON left_table.id = right_table.id"#
    );

    Ok(())
}

/// The clause a filter belongs in depends on the join type, but a limited input
/// is always its own subquery — including on a side whose filter would
/// otherwise have to stay in `ON`, and on a side of a `FULL JOIN`.
#[test]
fn test_outer_join_input_with_pushed_down_fetch() -> Result<()> {
    let schema_left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let schema_right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let mut rendered = String::new();
    for join_type in [
        datafusion_expr::JoinType::Left,
        datafusion_expr::JoinType::Right,
        datafusion_expr::JoinType::Full,
    ] {
        let plan = LogicalPlanBuilder::from(
            table_scan(Some("left_table"), &schema_left, Some(vec![0, 1]))?.build()?,
        )
        .join(
            table_scan_with_filter_and_fetch(
                Some("right_table"),
                &schema_right,
                Some(vec![0, 1]),
                vec![col("right_table.age").gt(lit(30))],
                Some(5),
            )?
            .build()?,
            join_type,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;

        rendered.push_str(&plan_to_sql(&plan)?.to_string());
        rendered.push('\n');
    }

    // In all three the filter and the limit are inside the subquery: nothing is
    // left for the enclosing `ON` or `WHERE` to re-apply after the limit.
    assert_snapshot!(
        rendered,
        @r#"
    SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table LEFT OUTER JOIN (SELECT right_table.id, right_table.age FROM right_table WHERE (right_table.age > 30) LIMIT 5) AS right_table ON left_table.id = right_table.id
    SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table RIGHT OUTER JOIN (SELECT right_table.id, right_table.age FROM right_table WHERE (right_table.age > 30) LIMIT 5) AS right_table ON left_table.id = right_table.id
    SELECT left_table.id, left_table."name", right_table.id, right_table.age FROM left_table FULL JOIN (SELECT right_table.id, right_table.age FROM right_table WHERE (right_table.age > 30) LIMIT 5) AS right_table ON left_table.id = right_table.id
    "#
    );

    Ok(())
}

/// An aliased scan keeps its alias, so the columns the enclosing query
/// references still resolve.
#[test]
fn test_aliased_join_input_with_pushed_down_fetch() -> Result<()> {
    let schema_left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let schema_right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let plan = LogicalPlanBuilder::from(
        table_scan_with_filter_and_fetch(
            Some("left_table"),
            &schema_left,
            Some(vec![0, 1]),
            vec![col("left_table.id").eq(lit("a"))],
            Some(5),
        )?
        .alias("l")?
        .build()?,
    )
    .join(
        table_scan(Some("right_table"), &schema_right, Some(vec![0, 1]))?.build()?,
        datafusion_expr::JoinType::Inner,
        (vec!["l.id"], vec!["right_table.id"]),
        None,
    )?
    .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT l.id, l."name", right_table.id, right_table.age FROM (SELECT l.id, l."name" FROM left_table AS l WHERE (l.id = 'a') LIMIT 5) AS l INNER JOIN right_table ON l.id = right_table.id"#
    );

    Ok(())
}

/// Text alone does not prove the limit survived: plan the generated SQL back
/// and check that the limit is still bound to the one input, above that scan
/// and below the join.
#[test]
fn test_join_input_fetch_survives_replanning() -> Result<()> {
    let schema_j1 = Schema::new(vec![
        Field::new("j1_id", DataType::Int32, false),
        Field::new("j1_string", DataType::Utf8, false),
    ]);
    let schema_j2 = Schema::new(vec![
        Field::new("j2_id", DataType::Int32, false),
        Field::new("j2_string", DataType::Utf8, false),
    ]);

    let plan = LogicalPlanBuilder::from(
        table_scan_with_filter_and_fetch(
            Some("j1"),
            &schema_j1,
            Some(vec![0, 1]),
            vec![col("j1.j1_id").gt(lit(1))],
            Some(5),
        )?
        .build()?,
    )
    .join(
        table_scan(Some("j2"), &schema_j2, Some(vec![0, 1]))?.build()?,
        datafusion_expr::JoinType::Inner,
        (vec!["j1.j1_id"], vec!["j2.j2_id"]),
        None,
    )?
    .build()?;

    let sql = plan_to_sql(&plan)?;
    let statement = Parser::new(&GenericDialect {})
        .try_with_sql(&sql.to_string())?
        .parse_statement()?;
    let context = MockContextProvider {
        state: MockSessionState::default(),
    };
    let replanned = SqlToRel::new(&context).sql_statement_to_plan(statement)?;

    assert_snapshot!(
        replanned,
        @r"
    Projection: j1.j1_id, j1.j1_string, j2.j2_id, j2.j2_string
      Inner Join:  Filter: j1.j1_id = j2.j2_id
        SubqueryAlias: j1
          Limit: skip=0, fetch=5
            Projection: j1.j1_id, j1.j1_string
              Filter: j1.j1_id > Int64(1)
                TableScan: j1
        TableScan: j2
    "
    );

    Ok(())
}

/// The scan applies its own filters before its `fetch`; a `Filter` node above
/// the scan applies afterwards. Both end up in the derived table, but they
/// cannot share one `SELECT` — `WHERE` is evaluated before `LIMIT`, so a
/// single scope would silently filter first.
#[test]
fn test_join_input_filter_above_pushed_down_fetch() -> Result<()> {
    let schema_left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("v", DataType::Int32, false),
    ]);
    let schema_right = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);

    let left = table_scan_with_filter_and_fetch(
        Some("left_table"),
        &schema_left,
        Some(vec![0, 1]),
        vec![col("left_table.v").gt(lit(1))],
        Some(5),
    )?
    .filter(col("left_table.id").eq(lit("a")))?
    .build()?;

    let plan = LogicalPlanBuilder::from(left)
        .join(
            table_scan(Some("right_table"), &schema_right, Some(vec![0, 1]))?.build()?,
            datafusion_expr::JoinType::Inner,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, left_table.v, right_table.id, right_table.age FROM (SELECT * FROM (SELECT left_table.id, left_table.v FROM left_table WHERE (left_table.v > 1) LIMIT 5) AS left_table WHERE (left_table.id = 'a')) AS left_table INNER JOIN right_table ON left_table.id = right_table.id"#
    );

    Ok(())
}

/// A scan filter may name a column the projection prunes. Rebuilding the input
/// keeps the original projection and lets the filter reference the wider source
/// schema, which is what SQL allows too.
#[test]
fn test_join_input_fetch_with_pruned_filter_column() -> Result<()> {
    let schema_left = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let schema_right = Schema::new(vec![Field::new("id", DataType::Utf8, false)]);

    let left = table_scan_with_filter_and_fetch(
        Some("left_table"),
        &schema_left,
        Some(vec![0]),
        vec![col("left_table.name").eq(lit("x"))],
        Some(5),
    )?
    .build()?;

    let plan = LogicalPlanBuilder::from(left)
        .join(
            table_scan(Some("right_table"), &schema_right, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            (vec!["left_table.id"], vec!["right_table.id"]),
            None,
        )?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT left_table.id, right_table.id FROM (SELECT left_table.id FROM left_table WHERE (left_table."name" = 'x') LIMIT 5) AS left_table INNER JOIN right_table ON left_table.id = right_table.id"#
    );

    Ok(())
}

#[test]
fn test_filter_on_unnamed_projection_output_binds() -> Result<()> {
    // A `Filter` whose predicate references a `Projection` output that the
    // projection does not name. The projected expression carries the logical
    // name `t.a + t.b`, which is not an identifier any relation exposes, so
    // emitting it as one produces SQL no engine can bind.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b"))])?
        .filter(col("t.a + t.b").gt(lit(1)))?
        .build()?;

    let sql = plan_to_sql(&plan)?;
    assert_snapshot!(sql, @r#"SELECT (t.a + t.b) FROM t WHERE ((t.a + t.b) > 1)"#);
    Ok(())
}

#[test]
fn test_filter_on_named_projection_output_is_unchanged() -> Result<()> {
    // An alias gives the output a name the emitted `SELECT` carries, and a bare
    // column keeps the name it already had. Neither needs repairing, so neither
    // is inlined.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);

    let aliased = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b")).alias("s")])?
        .filter(col("s").gt(lit(1)))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&aliased)?,
        @r#"SELECT (t.a + t.b) AS s FROM t WHERE (s > 1)"#
    );

    let bare_column = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a")])?
        .filter(col("t.a").gt(lit(1)))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&bare_column)?,
        @r#"SELECT t.a FROM t WHERE (t.a > 1)"#
    );
    Ok(())
}

#[test]
fn test_filter_on_unnamed_projection_output_used_twice() -> Result<()> {
    // Every reference to the output is inlined, not just the first.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b"))])?
        .filter(
            col("t.a + t.b")
                .gt(lit(1))
                .and(col("t.a + t.b").lt(lit(10))),
        )?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT (t.a + t.b) FROM t WHERE (((t.a + t.b) > 1) AND ((t.a + t.b) < 10))"#
    );
    Ok(())
}

#[test]
fn test_stacked_filters_on_unnamed_projection_output() -> Result<()> {
    // Stacked filters collapse into one `WHERE`, so the projection is still the
    // one this predicate refers to and both predicates are repaired.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b"))])?
        .filter(col("t.a + t.b").gt(lit(1)))?
        .filter(col("t.a + t.b").lt(lit(10)))?
        .build()?;
    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT (t.a + t.b) FROM t WHERE ((t.a + t.b) < 10) AND ((t.a + t.b) > 1)"#
    );
    Ok(())
}

#[test]
fn test_filter_on_unnamed_volatile_projection_output_is_not_inlined() -> Result<()> {
    // Inlining a volatile expression would evaluate it a second time, in a
    // clause that can see a different value than the `SELECT` list did, turning
    // an unbindable reference into silently wrong rows. The unbindable
    // reference is the safer of the two, so it is left in place.
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0]))?
        .project(vec![datafusion_functions::math::random().call(vec![])])?
        .filter(col("random()").gt(lit(0.5)))?
        .build()?;

    let sql = plan_to_sql(&plan)?;
    assert_snapshot!(sql, @r#"SELECT random() FROM t WHERE ("random()" > 0.5)"#);
    Ok(())
}

#[test]
fn test_derived_projection_output_is_named() -> Result<()> {
    // The `Filter` sits above the projection rather than beside it, so the
    // projection becomes a derived table and the predicate keeps the reference
    // instead of inlining what produces it. The reference is the output's logical
    // name, which the derived table only carries once the projection names it.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b"))])?
        .filter(col("t.a + t.b").gt(lit(1)))?
        .project(vec![col("t.a + t.b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "t.a + t.b" FROM (SELECT (t.a + t.b) AS "t.a + t.b" FROM t) WHERE ("t.a + t.b" > 1)"#
    );
    Ok(())
}

#[test]
fn test_derived_limit_output_is_named() -> Result<()> {
    // A row limit puts the projection in a scope of its own for a reason of its
    // own, and the outer reference is unbindable there for the same reason.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b"))])?
        .limit(0, Some(5))?
        .project(vec![col("t.a + t.b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "t.a + t.b" FROM (SELECT (t.a + t.b) AS "t.a + t.b" FROM t LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_derived_literal_output_is_named() -> Result<()> {
    // A literal is named by the engine too, so a reference to one from the
    // enclosing scope needs the same repair.
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0]))?
        .project(vec![lit(1)])?
        .filter(col("Int32(1)").gt(lit(0)))?
        .project(vec![col("Int32(1)")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "Int32(1)" FROM (SELECT 1 AS "Int32(1)" FROM t) WHERE ("Int32(1)" > 0)"#
    );
    Ok(())
}

#[test]
fn test_derived_volatile_output_is_named_and_evaluated_once() -> Result<()> {
    // A volatile output is left in place rather than inlined, since inlining
    // evaluates it a second time and answers the query with rows the `SELECT`
    // list never saw. Naming the output binds the reference *and* keeps the
    // single evaluation, so the scope the derived table provides repairs the
    // case that inlining cannot.
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0]))?
        .project(vec![datafusion_functions::math::random().call(vec![])])?
        .filter(col("random()").gt(lit(0.5)))?
        .project(vec![col("random()")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "random()" FROM (SELECT random() AS "random()" FROM t) WHERE ("random()" > 0.5)"#
    );
    Ok(())
}

#[test]
fn test_derived_output_under_sort_and_distinct_is_named() -> Result<()> {
    // The nodes between the derived table and the projection carry the output's
    // name out unchanged, so the projection below is still the one that has to
    // name it — including for the `ORDER BY` inside the derived table, which
    // refers to the output by the same name the enclosing scope does.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b"))])?
        .distinct()?
        .sort(vec![col("t.a + t.b").sort(true, false)])?
        .limit(0, Some(5))?
        .project(vec![col("t.a + t.b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "t.a + t.b" FROM (SELECT DISTINCT (t.a + t.b) AS "t.a + t.b" FROM t ORDER BY "t.a + t.b" ASC NULLS LAST LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_derived_output_under_subquery_alias_is_named() -> Result<()> {
    // The subquery alias renames the relation, not the output, so the name the
    // enclosing scope uses is still the one the projection reports — and the
    // requalification of the expression onto the alias happens after the output
    // has been named, so the two do not have to agree on the expression's spelling.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = LogicalPlanBuilder::from(
        table_scan(Some("t"), &schema, Some(vec![0, 1]))?
            .project(vec![col("t.a").add(col("t.b"))])?
            .alias("s")?
            .build()?,
    )
    .limit(0, Some(5))?
    .project(vec![Expr::Column(Column::new(
        Some(TableReference::bare("s")),
        "t.a + t.b",
    ))])?
    .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "t.a + t.b" FROM (SELECT (s.a + s.b) AS "t.a + t.b" FROM (SELECT s.a, s.b FROM t AS s) AS s LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_derived_output_under_distinct_on_with_sort_is_named() -> Result<()> {
    // `DISTINCT ON` carries its own `ORDER BY`, and rebuilding the node from the
    // expressions a plan reports drops it — `LogicalPlan::with_new_exprs` asserts
    // it was not asked to. Replacing only the input keeps every other field, so a
    // plan with sort expressions is named rather than panicked on.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b")), col("t.a")])?
        .distinct_on(
            vec![col("t.a")],
            vec![col("t.a + t.b"), col("t.a")],
            Some(vec![col("t.a").sort(true, false)]),
        )?
        .limit(0, Some(5))?
        .project(vec![col("t.a + t.b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "t.a + t.b" FROM (SELECT DISTINCT ON (t.a) "t.a + t.b", a FROM (SELECT (t.a + t.b) AS "t.a + t.b", t.a FROM t) ORDER BY a ASC NULLS LAST LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_derived_computed_distinct_on_output_is_named() -> Result<()> {
    // A `DISTINCT ON` emits its own `select_expr` as the `SELECT` list, so a
    // computed one is an unnamed output of the derived table even with no
    // projection beneath to name — the projection under a `DISTINCT ON` is
    // flattened into the same `SELECT` rather than exposed as a scope.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .distinct_on(
            vec![col("t.a")],
            vec![col("t.a").add(col("t.b"))],
            Some(vec![col("t.a").sort(true, false)]),
        )?
        .limit(0, Some(5))?
        .project(vec![col("t.a + t.b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "t.a + t.b" FROM (SELECT DISTINCT ON (t.a) (t.a + t.b) AS "t.a + t.b" FROM t ORDER BY t.a ASC NULLS LAST LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_distinct_on_output_is_not_named_over_its_own_key() -> Result<()> {
    // Naming an output must not change what another clause resolves to. PostgreSQL
    // resolves a bare name in `DISTINCT ON` and `ORDER BY` against the output list
    // before the input columns, so aliasing this output to "a + b" would rebind
    // `DISTINCT ON ("a + b")` from the input column (t.c) to the sum — the same rows
    // grouped by a different key. Leaving it unnamed keeps the enclosing reference
    // unbound, which is the pre-existing bug rather than a new wrong answer. Fixing
    // both needs the key emitted unambiguously: spiceai/spiceai#13444.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1, 2]))?
        .project(vec![
            col("t.a").alias("a"),
            col("t.b").alias("b"),
            col("t.c").alias("a + b"),
        ])?
        .distinct_on(
            vec![col("a + b")],
            vec![col("a").add(col("b"))],
            Some(vec![col("a + b").sort(true, false)]),
        )?
        .limit(0, Some(5))?
        .project(vec![col("a + b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "a + b" FROM (SELECT DISTINCT ON ("a + b") (a + b) FROM (SELECT t.a AS a, t.b AS b, t.c AS "a + b" FROM t) ORDER BY "a + b" ASC NULLS LAST LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_distinct_on_output_is_not_named_over_a_case_folded_key() -> Result<()> {
    // The same capture, reached by a spelling the key does not match byte-for-byte.
    // Quoting does not say how the engine compares what was written — DuckDB matches
    // identifiers case-insensitively even quoted — so naming this output "a + b" over a
    // key spelled "A + B" rebinds `DISTINCT ON ("A + B")` there just as an exact match
    // would. The comparison folds for that reason; on a case-sensitive dialect it
    // over-refuses, which costs an unbound reference rather than rows.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
    ]);
    // Built rather than parsed: `col()` lowercases an unquoted identifier, which would
    // erase the very case difference under test.
    let key = Expr::Column(Column::new_unqualified("A + B"));
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1, 2]))?
        .project(vec![
            col("t.a").alias("a"),
            col("t.b").alias("b"),
            col("t.c").alias("A + B"),
        ])?
        .distinct_on(
            vec![key.clone()],
            vec![col("a").add(col("b"))],
            Some(vec![key.clone().sort(true, false)]),
        )?
        .limit(0, Some(5))?
        // The enclosing projection is what makes the `DISTINCT ON` a derived table whose
        // outputs another scope binds, which is the only shape the naming pass runs on. It
        // binds the sum by the name the node's schema gives it — the very name an alias
        // would spell, and the one a case-folding engine matches the key against.
        .project(vec![Expr::Column(Column::new_unqualified("a + b"))])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "a + b" FROM (SELECT DISTINCT ON ("A + B") (a + b) FROM (SELECT t.a AS a, t.b AS b, t.c AS "A + B" FROM t) ORDER BY "A + B" ASC NULLS LAST LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_distinct_on_output_is_named_over_a_qualified_key() -> Result<()> {
    // The counterpart to the test above, and the assumption that keeps its skip
    // narrow: only a *bare* key can be captured by an output alias. A qualified one
    // survives emission as `d."a + b"`, which resolves to the relation in both
    // clauses whatever the output list holds — so naming still applies here.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
    ]);
    let key = Expr::Column(Column::new(Some(TableReference::bare("d")), "a + b"));
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1, 2]))?
        .project(vec![
            col("t.a").alias("a"),
            col("t.b").alias("b"),
            col("t.c").alias("a + b"),
        ])?
        .alias("d")?
        .distinct_on(
            vec![key.clone()],
            vec![col("d.a").add(col("d.b"))],
            Some(vec![key.clone().sort(true, false)]),
        )?
        .limit(0, Some(5))?
        .project(vec![col("d.a + d.b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "d.a + d.b" FROM (SELECT DISTINCT ON (d."a + b") (d.a + d.b) AS "d.a + d.b" FROM (SELECT d.a AS a, d.b AS b, d.c AS "a + b" FROM t AS d) AS d ORDER BY d."a + b" ASC NULLS LAST LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_distinct_on_output_is_not_named_over_a_correlated_key() -> Result<()> {
    // The same capture, reached by the other expression that emits a bare name. A
    // correlated key is an `Expr::OuterReferenceColumn`, which unparses through the
    // very same `col_to_sql` as `Expr::Column` and so emits `"a + b"` with nothing to
    // distinguish it — leaving the enclosing scope's resolution just as free to bind
    // the key to an output aliased that way. Reserving only `Expr::Column` would let
    // the alias back in through this variant.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
    ]);
    let key = Expr::OuterReferenceColumn(
        Arc::new(Field::new("a + b", DataType::Int32, false)),
        Column::new_unqualified("a + b"),
    );
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1, 2]))?
        .project(vec![
            col("t.a").alias("a"),
            col("t.b").alias("b"),
            col("t.c").alias("a + b"),
        ])?
        .distinct_on(
            vec![key.clone()],
            vec![col("a").add(col("b"))],
            Some(vec![key.clone().sort(true, false)]),
        )?
        .limit(0, Some(5))?
        // As in the sibling tests, this enclosing projection is what makes the
        // `DISTINCT ON` a derived table, which is the only shape the naming pass runs
        // on at all.
        .project(vec![Expr::Column(Column::new_unqualified("a + b"))])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "a + b" FROM (SELECT DISTINCT ON ("a + b") (a + b) FROM (SELECT t.a AS a, t.b AS b, t.c AS "a + b" FROM t) ORDER BY "a + b" ASC NULLS LAST LIMIT 5)"#
    );
    Ok(())
}

#[test]
fn test_named_derived_projection_outputs_are_unchanged() -> Result<()> {
    // An alias and a bare column already carry the name the enclosing scope
    // uses, so neither is renamed and neither picks up a redundant alias.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = table_scan(Some("t"), &schema, Some(vec![0, 1]))?
        .project(vec![col("t.a").add(col("t.b")).alias("s"), col("t.b")])?
        .filter(col("s").gt(lit(1)))?
        .project(vec![col("s"), col("t.b")])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT s, b FROM (SELECT (t.a + t.b) AS s, t.b FROM t) WHERE (s > 1)"#
    );
    Ok(())
}

#[test]
fn test_derived_output_name_carrying_a_quote_is_escaped() -> Result<()> {
    // The name comes from the plan, so it can carry any character a literal or a
    // column name can — including the quote that delimits the identifier it is
    // emitted as. Both the alias and the reference to it escape it, so neither
    // closes the identifier early and injects into the statement.
    let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]);
    let hostile = r#"Utf8("x" OR 1=1 --")"#;
    let plan = table_scan(Some("t"), &schema, Some(vec![0]))?
        .project(vec![lit("x\" OR 1=1 --")])?
        .filter(col(hostile).is_not_null())?
        .project(vec![col(hostile)])?
        .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT "Utf8(""x"" OR 1=1 --"")" FROM (SELECT 'x" OR 1=1 --' AS "Utf8(""x"" OR 1=1 --"")" FROM t) WHERE "Utf8(""x"" OR 1=1 --"")" IS NOT NULL"#
    );
    Ok(())
}

#[test]
fn test_subquery_alias_over_pushed_down_scan_still_unbindable() -> Result<()> {
    // The scan pushdown requalifies the projection onto the subquery alias before
    // the derived table is built, so the name the derived table can report for the
    // output — `s.a + s.b` — is not the one the enclosing scope uses for it,
    // `t.a + t.b`. Naming the output cannot close that gap: the repair is for the
    // enclosing scope to name the relation's columns on the alias it attaches.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let plan = LogicalPlanBuilder::from(
        table_scan(Some("t"), &schema, Some(vec![0, 1]))?
            .project(vec![col("t.a").add(col("t.b"))])?
            .alias("s")?
            .build()?,
    )
    .project(vec![Expr::Column(Column::new(
        Some(TableReference::bare("s")),
        "t.a + t.b",
    ))])?
    .build()?;

    assert_snapshot!(
        plan_to_sql(&plan)?,
        @r#"SELECT s."t.a + t.b" FROM (SELECT (s.a + s.b) AS "s.a + s.b" FROM t AS s) AS s"#
    );
    Ok(())
}

#[test]
fn test_qualified_join_input_fetch_refused_on_full_qualified_col_dialect() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
    ]);
    let other = Schema::new(vec![Field::new("id", DataType::Utf8, false)]);

    let join_with_fetched_input = |name: &str| -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(
            table_scan_with_filter_and_fetch(
                Some(name),
                &schema,
                Some(vec![0, 1]),
                vec![],
                Some(5),
            )?
            .build()?,
        )
        .join(
            table_scan(Some("other"), &other, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            (vec![format!("{name}.id")], vec!["other.id".to_string()]),
            None,
        )?
        .build()
    };

    let dialect = CustomDialectBuilder::default()
        .with_full_qualified_col(true)
        .build();
    let unparser = Unparser::new(&dialect);

    // The derived table can only be aliased `table`, while this dialect spells
    // the join condition `catalog.schema.table.id` — a relation the derived
    // table no longer brings into scope.
    let err = unparser
        .plan_to_sql(&join_with_fetched_input("catalog.schema.table")?)
        .expect_err("a qualified name must not unparse to an unresolvable alias");
    assert_contains!(
        err.to_string(),
        "not supported for a qualified table name on a dialect that spells columns in full"
    );

    // A single-component name loses nothing to the alias, so the guard must
    // not fire on it even on this dialect.
    assert_snapshot!(
        unparser.plan_to_sql(&join_with_fetched_input("t")?)?,
        @"SELECT t.id, t.value, other.id FROM (SELECT t.id, t.value FROM t LIMIT 5) AS t INNER JOIN other ON t.id = other.id"
    );
    Ok(())
}

/// `LeftSemi Join` whose equi-join key carries a reference reaching past the
/// join, on the half named by `outer_ref_side`.
///
/// Built through `join_with_expr_keys` because that is the only route that
/// admits one: `find_valid_equijoin_key_pair` decides which input a key belongs
/// to from `Expr::column_refs`, which collects `Expr::Column` alone, so
/// `p.x + outer_ref(p.c)` reads as belonging to `p` with the outer reference
/// never examined. Each key pairs a local column of its own side with the outer
/// reference, since a key of only outer references owns no input and is routed
/// to the filter instead.
fn outer_reference_in_join_key(
    outer_ref_side: OnHalf,
    outer_ref: &str,
) -> Result<LogicalPlan> {
    let schema = int32_schema(&["x", "c"]);
    let probe = table_scan(Some("p"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("b"), &schema, Some(vec![0, 1]))?.build()?;
    let outer = out_ref_col(DataType::Int32, outer_ref);
    let (left, right) = match outer_ref_side {
        OnHalf::Probe => (col("p.x") + outer, col("b.x")),
        OnHalf::Build => (col("p.x"), col("b.x") + outer),
    };
    LogicalPlanBuilder::from(probe)
        .join_with_expr_keys(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (vec![left], vec![right]),
            None,
        )?
        .build()
}

/// Which half of an `on` pair a test puts its outer reference on.
///
/// Not `JoinSide` — `datafusion_common::JoinSide` is a different, public thing.
enum OnHalf {
    Probe,
    Build,
}

/// A reference reaching past the join, in the build half of an `on` pair, binds
/// to the body's own `FROM`.
///
/// The build half is not the correlated one, so the side split skips it — right
/// for an `Expr::Column`, which belongs to that input on purpose, and wrong for
/// an outer reference, which belongs to neither and is only passing through.
/// Before this was caught, the plan unparsed to
/// `SELECT "p"."x", "p"."c" FROM "p" WHERE EXISTS (SELECT 1 FROM "b" WHERE ("p"."x" = ("b"."x" + "b"."c")))`,
/// where the `out_ref_col("b.c")` written for an enclosing query binds to the
/// subquery's own `b` instead.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_in_build_half_of_a_key()
-> Result<()> {
    let plan = outer_reference_in_join_key(OnHalf::Build, "b.c")?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference the body's own FROM answers to must be refused",
    );
    Ok(())
}

/// The same reference in the probe half, captured by the *probe's* emitted
/// `FROM` rather than the build's.
///
/// `out_ref_col("p.c")` is on its way past the outer query as well, so the
/// probe shadowing it is as much a capture as the build shadowing it — the
/// build side here answers to nothing it names. Before this was caught, the
/// plan unparsed to
/// `SELECT "p"."x", "p"."c" FROM "p" WHERE EXISTS (SELECT 1 FROM "b" WHERE (("p"."x" + "p"."c") = "b"."x"))`,
/// where the `"p"."x"` correlation binds outward correctly and the `"p"."c"`
/// beside it stops one scope short of where it was written to reach.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_in_probe_half_of_a_key()
-> Result<()> {
    let plan = outer_reference_in_join_key(OnHalf::Probe, "p.c")?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference the probe's own FROM answers to must be refused",
    );
    Ok(())
}

/// The bound on the fix: asking both halves is for outer references only, and
/// an ordinary `Expr::Column` keeps the side split.
///
/// An outer reference naming a relation neither side emits is not captured, so
/// this plan has to unparse — and it unparses with the local `p.x`/`b.x`
/// columns in the same pair, which is what would break if the both-halves pass
/// applied the `Expr::Column` rule to them as well.
#[test]
fn test_unparse_left_semi_join_keeps_outer_reference_neither_side_answers_to()
-> Result<()> {
    let plan = outer_reference_in_join_key(OnHalf::Build, "elsewhere.c")?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(
        unparser.plan_to_sql(&plan)?,
        @r#"SELECT "p"."x", "p"."c" FROM "p" WHERE EXISTS (SELECT 1 FROM "b" WHERE ("p"."x" = ("b"."x" + "elsewhere"."c")))"#
    );
    Ok(())
}

/// The narrowing to outer references is what keeps the side split, and this is
/// the shape that needs it: one `on` half carrying an outer reference *and* a
/// plain column that both sides answer to.
///
/// The probe is `t1 INNER JOIN t` and the build side is `t INNER JOIN t3`, so
/// both emitted `FROM`s answer to `t`. The build half of the key is
/// `t.c + outer_ref(elsewhere.c)`: the `t.c` is an ordinary build-side column
/// binding inside on purpose, and the outer reference names a relation neither
/// side emits. Asking this half about *every* reference rather than only the
/// ones reaching past the join applies the `Expr::Column` rule to `t.c`, which
/// both scopes answer to, and refuses a plan that unparses correctly.
///
/// So the pass over both halves has to be narrowed by reference kind and not
/// merely by which halves it reads — a distinction no plan without both kinds
/// in one half can show.
#[test]
fn test_unparse_left_semi_join_keeps_shared_relation_column_beside_an_outer_reference()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("t1"), &schema, Some(vec![0, 1]))?
        .join_on(
            table_scan(Some("t"), &schema, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            vec![col("t1.d").eq(col("t.c"))],
        )?
        .build()?;
    let build = table_scan(Some("t"), &schema, Some(vec![0]))?
        .join_on(
            table_scan(Some("t3"), &schema, Some(vec![0]))?.build()?,
            datafusion_expr::JoinType::Inner,
            vec![col("t.c").eq(col("t3.c"))],
        )?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t1.c"), col("t1.d")])?
        .join_with_expr_keys(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (
                vec![col("t1.c")],
                vec![col("t.c") + out_ref_col(DataType::Int32, "elsewhere.c")],
            ),
            None,
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// A `LeftSemi` join whose build side is a `UNNEST` planned by `SqlToRel`,
/// correlated by an unqualified reference named `VALUE`.
///
/// The build side has to come from `SqlToRel` rather than `LogicalPlanBuilder`:
/// the `LATERAL FLATTEN` path fires only for a projection carrying a
/// `__unnest_placeholder` column, which `RecursiveUnnestRewriter` builds during
/// SQL planning. A hand-built `Unnest` takes a different path and would pin
/// nothing.
///
/// The correlation lives in `join.filter` and names only the probe, so every
/// identifier the body emits binds — there is no dangling build-half reference
/// to mask the capture behind a statement that would fail anyway.
fn unnest_build_side_correlated_by(correlation: Expr) -> Result<LogicalPlan> {
    let statement = Parser::new(&GenericDialect {})
        .try_with_sql("SELECT UNNEST([1,2,3])")?
        .parse_statement()?;
    let state = MockSessionState::default()
        .with_scalar_function(make_array_udf())
        .with_expr_planner(Arc::new(NestedFunctionPlanner))
        .with_expr_planner(Arc::new(FieldAccessPlanner))
        .with_expr_planner(Arc::new(CoreFunctionPlanner::default()));
    let build =
        SqlToRel::new(&MockContextProvider { state }).sql_statement_to_plan(statement)?;

    let probe_schema = Schema::new(vec![
        Field::new("VALUE", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ]);
    let probe = table_scan(Some("p"), &probe_schema, Some(vec![0, 1]))?
        .project(vec![
            Expr::Column(Column::new(Some(TableReference::bare("p")), "VALUE"))
                .alias("VALUE"),
            col("p.d").alias("d"),
        ])?
        .build()?;

    LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![correlation],
        )?
        .build()
}

/// A Snowflake `LATERAL FLATTEN` relation exposes column names the plan does
/// not hold, so the scope cannot be read off the plan at all.
///
/// `FLATTEN` presents its output as `VALUE` (and Snowflake's other FLATTEN
/// columns), while the plan carries the unnest under its own generated name.
/// Nothing a walk collects can contain `VALUE`, so an unqualified correlation on
/// that name passes every check and then binds to the FLATTEN. Before this was
/// caught, the plan unparsed to
/// `SELECT "p"."VALUE" AS "VALUE", "p"."d" AS "d" FROM "p" WHERE EXISTS (SELECT 1 FROM LATERAL FLATTEN(INPUT => [1, 2, 3]) AS "_unnest_1" WHERE ("VALUE" > 0))`,
/// where `("VALUE" > 0)` was written against the probe's `p.VALUE` and binds to
/// `_unnest_1.VALUE` instead. Every element of `[1, 2, 3]` is greater than zero,
/// so the `EXISTS` is unconditionally true and the semi join keeps every probe
/// row.
///
/// The *qualifier* `_unnest_1` was already covered, by the invented-alias list.
/// It is the column names that are not, and listing them here would mean
/// writing Snowflake's FLATTEN output schema into this walk — under-refusing the
/// moment that schema grows. So the scope says it cannot be read.
#[test]
fn test_unparse_left_semi_join_refuses_correlation_captured_by_a_flatten_relation()
-> Result<()> {
    let plan = unnest_build_side_correlated_by(
        Expr::Column(Column::new(None::<TableReference>, "VALUE")).gt(lit(0)),
    )?;

    assert_captured_correlation_refused_by(
        &Unparser::new(&SnowflakeDialect::new()),
        &plan,
        "a correlation the FLATTEN relation's own columns answer to must be refused",
    );
    Ok(())
}

/// The other direction: the same plan on a dialect that emits `UNNEST` as a bare
/// table factor keeps its pushdown.
///
/// BigQuery's `UNNEST([...])` introduces no name and no `VALUE` column, so the
/// correlation has nothing in the body to collide with and binds outward as
/// written. That is why the refusal is keyed on the dialect emitting `FLATTEN`
/// rather than on the `Unnest` node alone.
#[test]
fn test_unparse_left_semi_join_keeps_unnest_build_side_without_flatten() -> Result<()> {
    let plan = unnest_build_side_correlated_by(
        Expr::Column(Column::new(None::<TableReference>, "VALUE")).gt(lit(0)),
    )?;

    let unparser = Unparser::new(&BigQueryDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// The bound on the FLATTEN refusal: it costs the *unqualified* references
/// only, and a qualified correlation over the same build side keeps its
/// pushdown.
///
/// `LATERAL FLATTEN(...) AS "_unnest_1"` answers to exactly one name, and the
/// emitter picks it — so a reference qualified by anything else cannot bind
/// there, and one qualified by `_unnest_1` is caught by the invented-alias list
/// instead. Only an unqualified reference is undecidable, because the FLATTEN's
/// column names are the part the plan does not hold.
///
/// So `p.VALUE` binds outward as written even though the body contains a
/// FLATTEN, and the scope is `Readable` with an unknown column list rather than
/// unreadable outright — a distinction that costs nothing to keep and every
/// qualified Snowflake semi join to lose.
#[test]
fn test_unparse_left_semi_join_keeps_qualified_correlation_over_a_flatten_relation()
-> Result<()> {
    let plan = unnest_build_side_correlated_by(
        Expr::Column(Column::new(Some(TableReference::bare("p")), "VALUE")).gt(lit(0)),
    )?;

    let snowflake = SnowflakeDialect::new();
    let unparser = Unparser::new(&snowflake);
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// A relation the user happened to name like an alias the unparser invents is
/// still a relation, and a build-local filter reference to it binds inside the
/// body on purpose.
///
/// `derived_limit` is one of the seven names the emitter uses for derived tables
/// of its own, so a reference qualified by it might be captured — but only if
/// the emitter actually introduces one *and* the reference was meant to reach
/// outward. Here `derived_limit` is the build relation itself and the reference
/// is in `join.filter`, which the probe does not answer to: it is the same
/// build-local reference `keeps_build_only_unqualified_filter_name` allows,
/// wearing a reserved name.
///
/// So the invented-alias answer counts as the build side answering and then
/// follows the ordinary attribution, rather than refusing outright. Refusing
/// here would fail the query, not fall back — see the summary — for the sole
/// reason that someone named a table after one of our internal aliases.
#[test]
fn test_unparse_left_semi_join_keeps_build_local_filter_on_a_reserved_relation_name()
-> Result<()> {
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("p"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("derived_limit"), &schema, Some(vec![0, 1]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("derived_limit.c").gt(lit(0))],
        )?
        .build()?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// The mirror of the test above, and what keeps the invented-alias answer
/// load-bearing rather than redundant once it stopped refusing on sight.
///
/// Here `derived_limit` is the **probe** relation, and the filter reference to it
/// has to reach outward. The build side answers to nothing it names, so the
/// ordinary attribution alone would allow it — while the emitter may well wrap
/// the build side as `derived_limit` directly in that reference's way, which is
/// a name no scope can see because it is in no plan. Counting the invented-alias
/// answer as the build side answering is what keeps this refused.
///
/// Together with the build-relation case, this is why the answer is folded into
/// `captured_by_build` rather than either returned early or dropped: returning
/// early refuses the build-local reference, and dropping it emits this one.
#[test]
fn test_unparse_left_semi_join_refuses_probe_only_reserved_name_in_a_filter() -> Result<()>
{
    let schema = exists_fetch_schema();
    let probe = table_scan(Some("derived_limit"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("b"), &schema, Some(vec![0, 1]))?.build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("derived_limit.c").gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a probe-side reference an invented alias could shadow must be refused",
    );
    Ok(())
}

/// `LeftSemi` join whose build side is `build` behind the alias `a`, correlated
/// by the unqualified reference `correlated_name`.
fn aliased_build_side_unqualified_correlation(
    build: LogicalPlan,
    correlated_name: &str,
) -> Result<LogicalPlan> {
    let probe = unqualified_probe()?;
    let build = subquery_alias(build, "a")?;
    LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![
                Expr::Column(Column::new(None::<TableReference>, correlated_name))
                    .gt(lit(0)),
            ],
        )?
        .build()
}

/// An alias replaces the names below it, not the columns, so the walk cannot
/// stop collecting columns where it stops collecting qualifiers.
///
/// The build side is `t` behind `AS "a"`, projected down to `x` alone, and the
/// correlation is an unqualified `c` — a column `t` has and the aliased plan
/// does not output. The emitter writes the side as an aliased *scan*, so
/// `FROM "t" AS "a"` answers to every column `t` has, `c` included. Before this
/// was caught, the plan unparsed to
/// `SELECT "p"."c" AS "c", "p"."d" AS "d" FROM "p" WHERE EXISTS (SELECT 1 FROM "t" AS "a" WHERE ("c" > 0))`,
/// where `("c" > 0)` was written against the probe's `p.c` and binds to `t.c`.
///
/// The alias is the entire difference: the same plan without it is refused,
/// because the scan's own schema is collected at the top level. That is what
/// makes this a boundary the walk was jumping rather than a missing arm — the
/// column names are gathered at the plan's root and at its leaf scans, and an
/// alias sits between the two.
#[test]
fn test_unparse_left_semi_join_refuses_unqualified_capture_behind_an_alias() -> Result<()>
{
    let inner = table_scan(Some("t"), &int32_schema(&["x", "c"]), Some(vec![0]))?
        .project(vec![col("t.x")])?
        .build()?;
    let plan = aliased_build_side_unqualified_correlation(inner, "c")?;

    assert_captured_correlation_refused(
        &plan,
        "a column the aliased relation exposes must be seen through the alias",
    );
    Ok(())
}

/// The other half of the same boundary: a name the aliased plan *renames into*
/// the body, which no relation below the alias has at all.
///
/// The alias deliberately is **not** the build plan's root — a rename above it
/// puts `c` out of the root schema, which is otherwise collected at the top
/// level and would cover this case for the wrong reason. `t` holds `x` and `y`;
/// the aliased side renames `x` to `c`; the projection above renames that to
/// `renamed`. So the emitted derived table answers to `c` while neither `t` nor
/// the build plan's output has it, and only the aliased subtree's own output
/// schema says so.
///
/// A first version of this test put the alias at the root, and the neuter that
/// drops the output-schema collection killed nothing — the top-level push was
/// carrying it. That is the whole reason for the shape here.
#[test]
fn test_unparse_left_semi_join_refuses_renamed_capture_behind_an_inner_alias()
-> Result<()> {
    let inner = table_scan(Some("t"), &int32_schema(&["x", "y"]), Some(vec![0]))?
        .project(vec![col("t.x").alias("c")])?
        .build()?;
    let build = LogicalPlanBuilder::from(subquery_alias(inner, "a")?)
        .project(vec![col("a.c").alias("renamed")])?
        .build()?;
    let plan = LogicalPlanBuilder::from(unqualified_probe()?)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![Expr::Column(Column::new(None::<TableReference>, "c")).gt(lit(0))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a name the aliased plan renames into the body must be seen through the alias",
    );
    Ok(())
}

/// And the direction that must keep its pushdown: an unqualified reference to a
/// name nothing behind the alias answers to, neither as a column of the scan nor
/// as an output of the aliased plan.
///
/// Without this the fix above could have been "refuse every unqualified
/// correlation whenever the build side carries an alias", which no other test
/// here would notice.
#[test]
fn test_unparse_left_semi_join_keeps_unqualified_name_absent_behind_an_alias()
-> Result<()> {
    let inner = table_scan(Some("t"), &int32_schema(&["x", "y"]), Some(vec![0, 1]))?
        .project(vec![col("t.x"), col("t.y")])?
        .build()?;
    let plan = aliased_build_side_unqualified_correlation(inner, "c")?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// A semi join from `public.t` to `other.t` whose filter is a nested `EXISTS`
/// carrying `outer_ref(<outer_relation>.c)`.
fn nested_exists_outer_reference(outer_relation: &str) -> Result<LogicalPlan> {
    nested_subquery_outer_reference(outer_relation, exists)
}

/// [`nested_exists_outer_reference`] for the other subquery-bearing expressions:
/// `wrap` decides which one holds the subquery, and the reference the subquery
/// carries is the same either way.
fn nested_subquery_outer_reference(
    outer_relation: &str,
    wrap: impl FnOnce(Arc<LogicalPlan>) -> Expr,
) -> Result<LogicalPlan> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let inner = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(
            col("z.k").eq(out_ref_col(DataType::Int32, format!("{outer_relation}.c"))),
        )?
        .build()?;

    LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![wrap(Arc::new(inner))],
        )?
        .build()
}

/// An outer reference held by a *nested* subquery is invisible to a walk over
/// the predicate, and it passes through the `EXISTS` body this join introduces.
///
/// `Expr::apply_children` returns `Continue` for `Exists` and `ScalarSubquery`
/// without descending, so the plan inside one is never reached and the outer
/// references it carries — held separately on `Subquery::outer_ref_columns` —
/// are seen by nothing. `Expr::contains_outer` reports `false` for this
/// predicate; the walk that looks for captures sees an empty set of references.
///
/// Here the reference names `public.t`, the probe. On a dialect that spells
/// neither qualifier in full the body introduces `FROM "other"."t"`, and the
/// nested `"t"."c"` resolves against the innermost scope answering to `t` —
/// which is now `other.t`. Before this was caught, the plan unparsed to
/// `SELECT "c", "d" FROM "public"."t" WHERE EXISTS (SELECT 1 FROM "other"."t" WHERE EXISTS (SELECT "z"."k" FROM "z" WHERE ("z"."k" = "t"."c")))`,
/// where the innermost comparison reads `other.t.c` against itself rather than
/// against the row the query was written to correlate with.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_held_by_a_nested_subquery()
-> Result<()> {
    let plan = nested_exists_outer_reference("public.t")?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference a nested subquery holds must be refused",
    );
    Ok(())
}

/// The other direction: a nested subquery whose outer reference names a relation
/// neither side of this join introduces keeps its pushdown.
///
/// `elsewhere.c` passes through both emitted scopes untouched, so there is
/// nothing for it to collide with and the correlation still reaches whatever
/// enclosing query it was written against. Without this, the fix above could
/// have been "refuse every join whose filter holds a subquery", which nothing
/// else here would catch.
#[test]
fn test_unparse_left_semi_join_keeps_nested_outer_reference_neither_side_answers_to()
-> Result<()> {
    let plan = nested_exists_outer_reference("elsewhere")?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// The case the pre-check is actually load-bearing for: a subquery-held outer
/// reference that only the **probe** shadows.
///
/// The build side is `b`, which answers to nothing the reference names, so the
/// probe-half pass — which asks the build scope alone — clears it. Only asking
/// the probe scope refuses it, and the probe scope is built only if this half
/// is noticed to carry such a reference. `Expr::contains_outer` says `false`
/// here, because the reference lives on the subquery.
#[test]
fn test_unparse_left_semi_join_refuses_probe_shadowed_reference_in_a_key_subquery()
-> Result<()> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(Some("p"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(Some("b"), &schema, Some(vec![0, 1]))?.build()?;
    let inner = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(col("z.k").eq(out_ref_col(DataType::Int32, "p.c")))?
        .project(vec![col("z.k")])?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join_with_expr_keys(
            build,
            datafusion_expr::JoinType::LeftSemi,
            (
                vec![col("p.c") + scalar_subquery(Arc::new(inner))],
                vec![col("b.c")],
            ),
            None,
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a subquery-held reference the probe's own FROM shadows must be refused",
    );
    Ok(())
}

/// The [`Subquery`] `scalar_subquery` builds for `plan`, whose
/// `outer_ref_columns` it populates.
///
/// [`Expr::SetComparison`] has no constructor function that does that, and a
/// hand-built `Subquery` with an empty `outer_ref_columns` would make the test
/// below pass for the wrong reason — there would be no reference to find.
///
/// [`Subquery`]: datafusion_expr::Subquery
fn populated_subquery(plan: Arc<LogicalPlan>) -> datafusion_expr::Subquery {
    match scalar_subquery(plan) {
        Expr::ScalarSubquery(subquery) => subquery,
        other => panic!("scalar_subquery built {other}"),
    }
}

/// `Expr::InSubquery` holds a subquery the same way `Exists` does, and the walk
/// reaches neither: `Expr::apply_children` descends into the compared
/// expression only, never into the subquery's plan.
///
/// The same shadowing as
/// `test_unparse_left_semi_join_refuses_outer_reference_held_by_a_nested_subquery`,
/// so what this adds is that the variant is answered for at all.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_held_by_a_nested_in_subquery()
-> Result<()> {
    let plan =
        nested_subquery_outer_reference("public.t", |inner| in_subquery(lit(1), inner))?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference an IN subquery holds must be refused",
    );
    Ok(())
}

/// `Expr::SetComparison` — `= ANY`, `> ALL` — is the fourth subquery-bearing
/// variant, and it is reached no more than the other three.
///
/// It is the one with no `expr_fn` constructor, so it is also the one most
/// easily left out of a match written from that module's exports.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_held_by_a_nested_set_comparison()
-> Result<()> {
    let plan = nested_subquery_outer_reference("public.t", |inner| {
        Expr::SetComparison(datafusion_expr::expr::SetComparison::new(
            Box::new(lit(1)),
            populated_subquery(inner),
            datafusion_expr::Operator::Gt,
            datafusion_expr::expr::SetQuantifier::Any,
        ))
    })?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference a set-comparison subquery holds must be refused",
    );
    Ok(())
}

/// A semi join from `public.t` to `other.t` whose filter is an `EXISTS` with a
/// further `EXISTS` nested inside it for each entry in `bodies`, the innermost
/// correlating on `outer_ref(<outer_relation>.c)`.
///
/// `bodies` names the relation each intermediate body selects from, outermost
/// first, which is how a test says where the correlation's qualifier is answered
/// — by one of those bodies, or by nothing until it reaches the join.
fn nested_exists_bodies(
    bodies: &[TableReference],
    outer_relation: &str,
) -> Result<LogicalPlan> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;

    let mut subquery = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(
            col("z.k").eq(out_ref_col(DataType::Int32, format!("{outer_relation}.c"))),
        )?
        .build()?;
    for body in bodies.iter().rev() {
        subquery = table_scan(Some(body.clone()), &schema, Some(vec![0, 1]))?
            .filter(exists(Arc::new(subquery)))?
            .build()?;
    }

    LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![exists(Arc::new(subquery))],
        )?
        .build()
}

/// An outer reference held two subqueries down is on no list the predicate's own
/// walk sees, and it still passes through the `EXISTS` body this join introduces.
///
/// Every site that builds [`Subquery::outer_ref_columns`] uses
/// `LogicalPlan::all_out_ref_exprs`, which collects over the plan's own
/// expressions and its `inputs()` — and `inputs()` excludes subqueries, while
/// `Expr` traversal reports a subquery-bearing expression as a leaf. So the
/// intermediate body's list is *empty*: reading it alone reports no reference to
/// examine, and the walk clears a plan it never looked at.
///
/// Here the correlation names `public.t`, the probe, and the body this join
/// emits introduces `FROM "other"."t"`. On a dialect that does not spell columns
/// in full both are written `"t"`, so the innermost `"t"."c"` binds to the
/// nearest enclosing scope answering to `t` — `other.t` — rather than to the row
/// the query was written to correlate with. Before this was caught, the plan
/// unparsed to
/// `SELECT "c", "d" FROM "public"."t" WHERE EXISTS (SELECT 1 FROM "other"."t" WHERE EXISTS (SELECT "w"."c", "w"."d" FROM "w" WHERE EXISTS (SELECT "z"."k" FROM "z" WHERE ("z"."k" = "t"."c"))))`,
/// which is valid SQL that runs and compares an inner row with itself.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_held_two_subqueries_down()
-> Result<()> {
    let plan = nested_exists_bodies(&[TableReference::bare("w")], "public.t")?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference held two subqueries down must be refused",
    );
    Ok(())
}

/// The other direction, and the reason the descent above cannot simply collect
/// every nested list: a nested subquery's outer references are relative to the
/// scope enclosing *it*, which is usually the body one level out rather than
/// anything past this join.
///
/// The correlation names `public.t` and the body holding it selects from
/// `public.t`, so `"t"."c"` binds to that body — exactly where the plan says it
/// should, and never reaching this join at all. A union of the nested lists
/// cannot tell this apart from the case above, and would refuse it; the refusal
/// is a hard error rather than a different emission, so it would trade wrong SQL
/// for a wrong error on a query that unparses correctly today.
#[test]
fn test_unparse_left_semi_join_keeps_nested_reference_the_enclosing_body_answers_to()
-> Result<()> {
    let plan =
        nested_exists_bodies(&[TableReference::partial("public", "t")], "public.t")?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// The descent is not two levels deep but every level: the same reference held
/// three subqueries down, past two bodies neither of which answers to it, is
/// still captured by the one this join emits.
#[test]
fn test_unparse_left_semi_join_refuses_outer_reference_held_three_subqueries_down()
-> Result<()> {
    let plan = nested_exists_bodies(
        &[TableReference::bare("w"), TableReference::bare("v")],
        "public.t",
    )?;

    assert_captured_correlation_refused(
        &plan,
        "an outer reference held three subqueries down must be refused",
    );
    Ok(())
}

/// And the bodies that answer are asked at every level too, not just the
/// outermost one.
///
/// `w` does not answer to `public.t`, but the body nested inside it does, so the
/// correlation binds there and this join never sees it. Testing only the
/// outermost body's scope would report the reference as still travelling and
/// refuse a plan that unparses correctly.
#[test]
fn test_unparse_left_semi_join_keeps_reference_an_inner_body_answers_to() -> Result<()> {
    let plan = nested_exists_bodies(
        &[
            TableReference::bare("w"),
            TableReference::partial("public", "t"),
        ],
        "public.t",
    )?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// The body that answers need not be the innermost one either: the scopes
/// between a reference and this join are asked as a chain, not one at a time.
///
/// `w` encloses the correlation and does not answer to it, and `public.t`
/// encloses `w` and does. The reference binds to that outer body, so it never
/// reaches this join — a chain rebuilt at each level from the enclosing body
/// alone would forget `public.t` and refuse it.
#[test]
fn test_unparse_left_semi_join_keeps_reference_an_outer_body_answers_to() -> Result<()> {
    let plan = nested_exists_bodies(
        &[
            TableReference::partial("public", "t"),
            TableReference::bare("w"),
        ],
        "public.t",
    )?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// A body whose emitted `FROM` the plan does not describe consumes *nothing*.
///
/// The scope built for such a body is [`EmittedScope::Unreadable`], which the
/// capture test reads as answering to every reference — the direction that
/// refuses there. Read the same way here it would do the opposite, because a
/// consumed reference is one that is never examined: the correlation would be
/// dropped on the strength of a name nobody knows, and the capture the join's own
/// scopes would have refused is emitted instead.
#[test]
fn test_unparse_left_semi_join_refuses_nested_reference_an_unreadable_body_cannot_place()
-> Result<()> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let innermost = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(col("z.k").eq(out_ref_col(DataType::Int32, "public.t.c")))?
        .build()?;
    let body = LogicalPlanBuilder::from(unqualified_extension(&["c", "d"])?)
        .filter(exists(Arc::new(innermost)))?
        .build()?;

    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![exists(Arc::new(body))],
        )?
        .build()?;

    assert_captured_correlation_refused_by(
        &extension_unparser("ext", None),
        &plan,
        "a nested reference an unreadable body cannot place must be refused",
    );
    Ok(())
}

/// A semi join from `public.t` to `other.t` whose filter is an `EXISTS` over a
/// body selecting from `a` joined to `joined`, correlating on
/// `outer_ref(public.t.c)` from a further `EXISTS` nested inside it.
///
/// `joined` is what a test varies: the reference's emitted qualifier collides
/// with the body's own `FROM` either way, and whether that collision is the
/// relation the plan named is the whole difference.
fn nested_exists_over_joined_body(joined: TableReference) -> Result<LogicalPlan> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let innermost = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(col("z.k").eq(out_ref_col(DataType::Int32, "public.t.c")))?
        .build()?;
    let body = LogicalPlanBuilder::from(
        table_scan(Some("a"), &schema, Some(vec![0, 1]))?.build()?,
    )
    .join_on(
        table_scan(Some(joined.clone()), &schema, Some(vec![0, 1]))?.build()?,
        datafusion_expr::JoinType::Inner,
        vec![col("a.c").eq(Expr::Column(Column::new(Some(joined), "c")))],
    )?
    .filter(exists(Arc::new(innermost)))?
    .build()?;

    LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![exists(Arc::new(body))],
        )?
        .build()
}

/// A body that answers to the reference without naming the relation it names
/// captures it, and the capture is decided before this join emits anything.
///
/// The correlation names `public.t`; the body selects from `mid.t`. On a dialect
/// that does not spell columns in full both are written `"t"`, so the emitted
/// `"t"."c"` resolves against `"mid"."t"` — an unrelated relation that merely
/// spells the same — and the comparison reads a row of `mid.t` against itself.
/// Reading a collision as the reference having arrived would drop it here
/// unexamined and emit
/// `... WHERE EXISTS (SELECT 1 FROM "other"."t" WHERE EXISTS (SELECT ... FROM "a" INNER JOIN "mid"."t" ON ("a"."c" = "t"."c") WHERE EXISTS (SELECT "z"."k" FROM "z" WHERE ("z"."k" = "t"."c"))))`,
/// which is valid SQL that runs and answers from that self-comparison.
#[test]
fn test_unparse_left_semi_join_refuses_nested_reference_a_body_captures_by_spelling()
-> Result<()> {
    let plan = nested_exists_over_joined_body(TableReference::partial("mid", "t"))?;

    assert_captured_correlation_refused(
        &plan,
        "a body colliding with the reference without naming its relation captures it",
    );
    Ok(())
}

/// The control the refusal above must not swallow: the same collision, by the
/// relation the plan actually named.
///
/// The body selects from `public.t`, which is where `outer_ref(public.t.c)` was
/// sent, so `"t"."c"` resolves exactly as the plan intends and this join never
/// sees the reference. Refusing every colliding body alike would refuse this too,
/// on SQL that binds correctly.
#[test]
fn test_unparse_left_semi_join_keeps_nested_reference_a_joined_body_names() -> Result<()>
{
    let plan = nested_exists_over_joined_body(TableReference::partial("public", "t"))?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// An `EXISTS` body that is a set operation, with the nested correlation in one
/// branch and `colliding` scanned by the other.
///
/// The branch holding the correlation selects from the very relation the
/// correlation names, so the reference binds there and never travels to the join.
fn nested_exists_over_union_body(colliding: TableReference) -> Result<LogicalPlan> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let innermost = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(col("z.k").eq(out_ref_col(DataType::Int32, "public.t.c")))?
        .build()?;
    let holding = table_scan(
        Some(TableReference::partial("public", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .filter(exists(Arc::new(innermost)))?
    .build()?;
    let body = LogicalPlanBuilder::from(holding)
        .union(table_scan(Some(colliding), &schema, Some(vec![0, 1]))?.build()?)?
        .build()?;

    LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![exists(Arc::new(body))],
        )?
        .build()
}

/// A set-operation body is emitted as one `SELECT` per branch, so a relation in
/// one branch is not in scope for a correlation held by another.
///
/// The branch holding the nested `EXISTS` selects from `public.t`, exactly where
/// `outer_ref(public.t.c)` was sent, so `"t"."c"` binds inside that branch and the
/// join never sees the reference. The sibling branch selects from `mid.t`, which
/// on this dialect is also written `"t"` — but it is emitted in a `SELECT` of its
/// own, whose `FROM` the first branch's `WHERE` cannot see. Scoping the two
/// branches as one body reads that sibling as a relation the reference could
/// resolve against and refuses SQL that binds correctly.
#[test]
fn test_unparse_left_semi_join_keeps_nested_reference_a_sibling_branch_names()
-> Result<()> {
    let plan = nested_exists_over_union_body(TableReference::partial("mid", "t"))?;

    let unparser = Unparser::new(&UnparserPostgreSqlDialect {});
    assert_snapshot!(unparser.plan_to_sql(&plan)?);
    Ok(())
}

/// The control the branch-scoping above must not swallow: a collision inside the
/// branch that holds the correlation is still a capture.
///
/// Splitting the body per branch narrows each scope, and narrowing it past the
/// branch's own `FROM` would stop reporting captures the emitter really does
/// write. Here the sibling branch names `public.t` while the holding branch scans
/// `mid.t`, so the nested `"t"."c"` resolves against that `mid.t` — a relation the
/// correlation does not name — and the comparison answers from a row of `mid.t`
/// read against itself.
#[test]
fn test_unparse_left_semi_join_refuses_nested_reference_its_own_branch_captures()
-> Result<()> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let innermost = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(col("z.k").eq(out_ref_col(DataType::Int32, "public.t.c")))?
        .build()?;
    let holding = table_scan(
        Some(TableReference::partial("mid", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .filter(exists(Arc::new(innermost)))?
    .build()?;
    let body = LogicalPlanBuilder::from(holding)
        .union(
            table_scan(
                Some(TableReference::partial("public", "t")),
                &schema,
                Some(vec![0, 1]),
            )?
            .build()?,
        )?
        .build()?;
    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![exists(Arc::new(body))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a branch colliding with the correlation it holds captures it",
    );
    Ok(())
}

/// The body's scope is the whole body, not the subtree beneath the node holding
/// the correlation — because that is the scope the *statement* has.
///
/// The nested `EXISTS` sits under a filter over `a` alone, with `mid.t` joined in
/// above it, so the subtree holding it knows only `a`. The emitter nonetheless
/// hoists the predicate to the body's own `WHERE`, emitting
/// `FROM "a" INNER JOIN "mid"."t" ON (…) WHERE EXISTS (…)`, where `"t"."c"`
/// resolves against `"mid"."t"` — a relation `far.t` is not. Neither side of this
/// join answers to the reference, so nothing downstream would refuse it: reading
/// the holding node's subtree as the scope reports a reference still travelling
/// outward and emits the capture.
#[test]
fn test_unparse_left_semi_join_refuses_nested_reference_captured_above_its_own_node()
-> Result<()> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "p")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "b")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let innermost = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(col("z.k").eq(out_ref_col(DataType::Int32, "far.t.c")))?
        .build()?;
    let body = LogicalPlanBuilder::from(
        table_scan(Some("a"), &schema, Some(vec![0, 1]))?
            .filter(exists(Arc::new(innermost)))?
            .build()?,
    )
    .join_on(
        table_scan(
            Some(TableReference::partial("mid", "t")),
            &schema,
            Some(vec![0, 1]),
        )?
        .build()?,
        datafusion_expr::JoinType::Inner,
        vec![col("a.c").eq(col("mid.t.c"))],
    )?
    .build()?;

    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![exists(Arc::new(body))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a reference the body captures above the node holding it must be refused",
    );
    Ok(())
}

/// A relation the emitter buries behind a derived table of its own is not one a
/// nested reference can be said to have arrived at.
///
/// The body is a `Window` over a `Projection` over `far.t`, which the emitter
/// renders as `FROM (SELECT "c", "d" FROM "far"."t") AS "derived_projection"` —
/// a fixed alias in place of whatever the plan called the relation. The plan
/// still says `far.t` plainly, so a scope collected from the plan alone reports
/// the reference as bound there and stops examining it; nothing after that would
/// look at it again, and this join's build side is `other.t`, whose emitted
/// component is the same `t` the reference is spelled with.
///
/// The refusal is a price, not a repair. Rendered, this plan's SQL happens to
/// bind correctly — the body's own `FROM` is aliased `"t"`, which re-supplies the
/// component the derived table took away, so the innermost `"t"."c"` resolves to
/// the row the correlation meant. Whether that holds is a property of aliases the
/// emitter chooses after this guard has run, so the guard cannot rely on it, and
/// being wrong the other way emits the capture instead of refusing it. This
/// records which way it errs.
#[test]
fn test_unparse_left_semi_join_refuses_nested_reference_behind_a_derived_window_input()
-> Result<()> {
    let schema = int32_schema(&["c", "d"]);
    let probe = table_scan(
        Some(TableReference::partial("public", "p")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let build = table_scan(
        Some(TableReference::partial("other", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .build()?;
    let innermost = table_scan(Some("z"), &int32_schema(&["k"]), Some(vec![0]))?
        .filter(col("z.k").eq(out_ref_col(DataType::Int32, "far.t.c")))?
        .build()?;

    let projected = table_scan(
        Some(TableReference::partial("far", "t")),
        &schema,
        Some(vec![0, 1]),
    )?
    .project(vec![col("far.t.c"), col("far.t.d")])?
    .build()?;
    let window_expr = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::AggregateUDF(sum_udaf()),
        params: WindowFunctionParams {
            args: vec![col("c")],
            partition_by: vec![],
            order_by: vec![],
            window_frame: WindowFrame::new(None),
            null_treatment: None,
            distinct: false,
            filter: None,
        },
    }));
    let body = LogicalPlanBuilder::from(projected)
        .window(vec![window_expr])?
        .filter(exists(Arc::new(innermost)))?
        .build()?;

    let plan = LogicalPlanBuilder::from(probe)
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![exists(Arc::new(body))],
        )?
        .build()?;

    assert_captured_correlation_refused(
        &plan,
        "a nested reference behind a derived table the emitter invents must not be \
         read as having arrived",
    );
    Ok(())
}

/// Builds a `LeftSemi` join whose build side is `SubqueryAlias(join_type(safe, t))`
/// and whose correlation is qualified by `t` — the relation the alias does *not*
/// rename.
///
/// The two joined relations are given disjoint column names on purpose. A shared
/// name makes the planner insert a renaming projection, and the alias is then
/// emitted as a derived table, which hides both relations and is the very case
/// this shape has to be kept distinct from.
fn aliased_join_build_side(join_on: Vec<Expr>) -> Result<LogicalPlan> {
    let probe =
        table_scan(Some("t"), &exists_fetch_schema(), Some(vec![0, 1]))?.build()?;

    let joined = table_scan(Some("t"), &int32_schema(&["c"]), Some(vec![0]))?.build()?;
    let inner = table_scan(Some("safe"), &int32_schema(&["k"]), Some(vec![0]))?
        .join_on(joined, datafusion_expr::JoinType::Inner, join_on)?
        .build()?;

    let build = LogicalPlan::SubqueryAlias(datafusion_expr::SubqueryAlias::try_new(
        Arc::new(inner),
        "a",
    )?);

    LogicalPlanBuilder::from(probe)
        .project(vec![col("t.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t.c").eq(col("a.k"))],
        )?
        .build()
}

/// An alias over a join renames one relation, not the join, so the relations
/// beside the renamed one stay addressable and can capture the correlation.
///
/// SQL has no syntax for naming a join, so an alias over one is emitted either as
/// a derived table — which does hide everything below it — or, for the join types
/// `requires_derived_subquery` declines to wrap, by renaming the primary relation
/// in place. This is the second shape:
///
/// ```sql
/// SELECT "t"."d" FROM "t"
/// WHERE EXISTS (SELECT 1 FROM "safe" AS "a" CROSS JOIN "t" WHERE ("t"."c" = "a"."k"))
/// ```
///
/// That is valid SQL, and every database runs it. `"t"."c"` was written against
/// the outer `"t"`, but the subquery's own `FROM` now introduces `"t"`, so it
/// binds there instead: the correlation is gone, the `EXISTS` asks only whether
/// the cross join has a row, and the semi join keeps probe rows it must drop.
/// Wrong rows, with nothing to report them.
///
/// Refused rather than repaired, on the same terms as every other capture this
/// guard finds.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_a_relation_beside_an_alias()
-> Result<()> {
    // No join condition, so this emits `CROSS JOIN` and nothing in the emitted
    // `ON` names the aliased relation: the SQL below is valid as well as wrong.
    let plan = aliased_join_build_side(vec![])?;

    assert_captured_correlation_refused(
        &plan,
        "a relation the build-side alias does not rename must be seen to capture",
    );
    Ok(())
}

/// The same shape with an inner join, so the refusal is not read as peculiar to
/// `CROSS JOIN`: what decides it is the alias having more than one relation to
/// replace, which is true of every join type `requires_derived_subquery` leaves
/// on the inline path.
#[test]
fn test_unparse_left_semi_join_refuses_capture_beside_an_aliased_inner_join() -> Result<()>
{
    let plan = aliased_join_build_side(vec![col("safe.k").eq(col("t.c"))])?;

    assert_captured_correlation_refused(
        &plan,
        "an inner join beside the aliased relation must be seen to capture too",
    );
    Ok(())
}

/// A schema-qualified `FROM` introduces the bare table name as a correlation
/// name, so it captures a reference qualified that way.
///
/// `FROM s.t2` is addressed as `t2` as readily as `s.t2` — that is how SQL
/// scopes a qualified relation, not a quirk of any one engine. Keying the scope
/// off the emitted qualifier alone looks only for `s.t2`, so a correlation
/// written against an *outer* `t2` matches nothing and reads as reaching
/// outward:
///
/// ```sql
/// SELECT t2.d FROM t2 WHERE EXISTS (SELECT 1 FROM s.t2 WHERE (t2.c = s.t2.c))
/// ```
///
/// Valid SQL, and wrong: `t2.c` binds to the subquery's own `s.t2`, so the
/// predicate compares the inner row with itself and the correlation is gone.
///
/// The full key is kept beside the bare one, so this does not merge `s1.t` with
/// `s2.t` — a correlation naming one of those still fails to match the other.
#[test]
fn test_unparse_left_semi_join_refuses_capture_by_a_qualified_scans_bare_name()
-> Result<()> {
    let schema = exists_fetch_schema();
    // The probe is named `t2`: the bare last component of the build's `s.t2`.
    let probe = table_scan(Some("t2"), &schema, Some(vec![0, 1]))?.build()?;
    let build = table_scan(
        Some(TableReference::partial("s", "t2")),
        &schema,
        Some(vec![0]),
    )?
    .build()?;

    let plan = LogicalPlanBuilder::from(probe)
        .project(vec![col("t2.d")])?
        .join_on(
            build,
            datafusion_expr::JoinType::LeftSemi,
            vec![col("t2.c").eq(col("s.t2.c"))],
        )?
        .build()?;

    assert_captured_correlation_refused_by(
        &Unparser::new(
            &CustomDialectBuilder::default()
                .with_full_qualified_col(true)
                .build(),
        ),
        &plan,
        "a qualified scan's bare name must be seen to capture",
    );
    Ok(())
}

/// A body spelled like the reference's relation but not identical to it is not
/// where the reference arrives, so crediting it drops the reference unexamined.
///
/// Deciding a reference *binds* to an enclosing body ends the enquiry into it —
/// the scopes past that one, which may capture it, are never asked. Every other
/// comparison in this guard folds case, because calling two identifiers the same
/// there costs a pushdown; folding here costs rows instead. On PostgreSQL a body
/// selecting from `"PUBLIC"."T"` does not resolve a reference written against
/// `"public"."t"`, but folded the two are one relation, and this was emitted:
///
/// ```sql
/// SELECT "c", "d" FROM "public"."t"
/// WHERE EXISTS (SELECT 1 FROM "other"."t"
///   WHERE EXISTS (SELECT "c", "d" FROM "PUBLIC"."T"
///     WHERE EXISTS (SELECT "z"."k" FROM "z" WHERE ("z"."k" = "t"."c"))))
/// ```
///
/// `"t"."c"` reaches no `"public"."t"` from there — it binds to the nearest
/// enclosing `FROM` that answers to `t`, which is a relation this join emits.
///
/// The exact comparison declines the arrival, so the reference goes on to be
/// examined and is refused as the capture it is. On a dialect that really does
/// bind the two alike the same strictness costs a pushdown, which is the trade
/// every other name in this guard already makes.
#[test]
fn test_unparse_left_semi_join_refuses_nested_reference_a_case_distinct_body_only_folds_onto()
-> Result<()> {
    let plan =
        nested_exists_bodies(&[TableReference::partial("PUBLIC", "T")], "public.t")?;

    assert_captured_correlation_refused(
        &plan,
        "a body that only case-folds onto the reference's relation must not be read as its home",
    );
    Ok(())
}

/// The `Projection` over `Aggregate` shape a grouped card plans to: group by a
/// truncated timestamp, then project a *wrapped* form of that same grouping
/// expression. The projection reads the aggregate's own output columns, whose names
/// come from the schema rather than being spelled here.
fn projection_wrapping_a_grouping_expr() -> Result<LogicalPlan> {
    let schema = Schema::new(vec![
        Field::new(
            "funded_ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
            true,
        ),
        Field::new("tok", DataType::Utf8, true),
    ]);
    let grouped = table_scan(Some("advances"), &schema, None)?
        .aggregate(
            vec![datafusion_functions::expr_fn::date_trunc(
                lit("week"),
                col("advances.funded_ts"),
            )],
            vec![count(lit(1i64))],
        )?
        .build()?;
    let mut outputs = grouped.schema().columns().into_iter();
    let group_output = outputs.next().expect("the grouping expression's output");
    let count_output = outputs.next().expect("the aggregate's output");
    LogicalPlanBuilder::from(grouped)
        .project(vec![
            cast(
                cast(Expr::Column(group_output), DataType::Date32),
                DataType::Utf8,
            )
            .alias("week_start"),
            Expr::Column(count_output).alias("advances_funded"),
        ])?
        .build()
}

/// A dialect that resolves `GROUP BY` against whole select items only gets the
/// aggregate in a scope of its own.
///
/// Flattened, the statement puts the grouping expression bare in `GROUP BY` and
/// wrapped in the select list, and `GoogleSQL` refuses it outright: "SELECT list
/// expression references column funded_ts which is neither grouped nor
/// aggregated". The whole statement fails, not one row of it.
///
/// The scope is asserted, not just that a statement comes out, because the two
/// cheaper renderings parse and return the wrong rows. `GROUP BY <output alias>`
/// and `GROUP BY <ordinal>` group by the value the projection computes, so a
/// projection that is not injective over the grouping expression collapses
/// distinct groups into one and sums their aggregates.
#[test]
fn test_projection_wrapping_a_grouping_expr_scopes_the_aggregate() -> Result<()> {
    let plan = projection_wrapping_a_grouping_expr()?;
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();

    let grouping_at = sql.find("GROUP BY").expect("a GROUP BY in: {sql}");
    let depth = sql[..grouping_at].chars().fold(0i32, |d, c| match c {
        '(' => d + 1,
        ')' => d - 1,
        _ => d,
    });
    assert!(
        depth >= 1,
        "the grouping has to be a scope of its own for this dialect: {sql}"
    );
    // The grouping expression must be rendered only inside the scope. Asserted on
    // the rendered call rather than on the base column, because the dialect
    // sanitises the derived output's alias out of the schema name, which spells the
    // base column inside it.
    let outer_select = &sql[..sql.find("FROM").expect("a FROM in: {sql}")];
    assert!(
        !outer_select.contains("TIMESTAMP_TRUNC"),
        "the outer select list still re-derives the grouping expression, which is what \
         this dialect cannot bind against its GROUP BY: {sql}"
    );
    // Non-vacuity: a scope that dropped the grouping or the aggregate would satisfy
    // both assertions above.
    assert!(
        sql[grouping_at..].contains("ISOWEEK") || sql.contains("TIMESTAMP_TRUNC"),
        "the scope has to carry the grouping expression: {sql}"
    );
    assert!(
        sql.to_uppercase().contains("COUNT("),
        "the scope has to carry the aggregate: {sql}"
    );
    Ok(())
}

/// A dialect that does resolve `GROUP BY` against sub-expressions keeps the single
/// `SELECT`, so the scope above is not paid where it buys nothing.
#[test]
fn test_projection_wrapping_a_grouping_expr_stays_flat_where_it_binds() -> Result<()> {
    let plan = projection_wrapping_a_grouping_expr()?;
    let sql = Unparser::new(&UnparserPostgreSqlDialect {})
        .plan_to_sql(&plan)?
        .to_string();

    let grouping_at = sql.find("GROUP BY").expect("a GROUP BY in: {sql}");
    let depth = sql[..grouping_at].chars().fold(0i32, |d, c| match c {
        '(' => d + 1,
        ')' => d - 1,
        _ => d,
    });
    assert_eq!(depth, 0, "PostgreSQL binds the wrapped select item: {sql}");
    Ok(())
}

/// A grouping expression that is a bare column needs no scope, in any dialect: a
/// column reference is matched wherever it appears. Without this the scope would be
/// paid on most grouped statements a `BigQuery` connector emits.
#[test]
fn test_projection_wrapping_a_grouped_column_stays_flat() -> Result<()> {
    let schema = Schema::new(vec![Field::new("tok", DataType::Utf8, true)]);
    let grouped = table_scan(Some("advances"), &schema, None)?
        .aggregate(vec![col("advances.tok")], vec![count(lit(1i64))])?
        .build()?;
    let mut outputs = grouped.schema().columns().into_iter();
    let group_output = outputs.next().expect("the grouped column's output");
    let count_output = outputs.next().expect("the aggregate's output");
    let plan = LogicalPlanBuilder::from(grouped)
        .project(vec![
            cast(Expr::Column(group_output), DataType::Utf8).alias("k"),
            Expr::Column(count_output).alias("n"),
        ])?
        .build()?;

    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    let grouping_at = sql.find("GROUP BY").expect("a GROUP BY in: {sql}");
    let depth = sql[..grouping_at].chars().fold(0i32, |d, c| match c {
        '(' => d + 1,
        ')' => d - 1,
        _ => d,
    });
    assert_eq!(
        depth, 0,
        "a grouped column reference is matched nested, so no scope is needed: {sql}"
    );
    Ok(())
}

/// A wrapped *aggregate* is legal in every dialect — `CAST(min(t) AS DATE)` is not
/// an ungrouped column reference — so it must not trigger the scope either.
#[test]
fn test_projection_wrapping_an_aggregate_stays_flat() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new(
            "funded_ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
            true,
        ),
        Field::new("tok", DataType::Utf8, true),
    ]);
    let grouped = table_scan(Some("advances"), &schema, None)?
        .aggregate(
            vec![col("advances.tok")],
            vec![datafusion_functions_aggregate::expr_fn::min(col(
                "advances.funded_ts",
            ))],
        )?
        .build()?;
    let mut outputs = grouped.schema().columns().into_iter();
    let group_output = outputs.next().expect("the grouped column's output");
    let min_output = outputs.next().expect("the aggregate's output");
    let plan = LogicalPlanBuilder::from(grouped)
        .project(vec![
            Expr::Column(group_output).alias("tok"),
            cast(Expr::Column(min_output), DataType::Date32).alias("first_day"),
        ])?
        .build()?;

    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    let grouping_at = sql.find("GROUP BY").expect("a GROUP BY in: {sql}");
    let depth = sql[..grouping_at].chars().fold(0i32, |d, c| match c {
        '(' => d + 1,
        ')' => d - 1,
        _ => d,
    });
    assert_eq!(depth, 0, "a wrapped aggregate needs no scope: {sql}");
    Ok(())
}

/// `SUM(v) OVER (ORDER BY k …)` with the frame `frame` and the NULL placement
/// `asc`/`nulls_first`. `k` is nullable, which is the case the placement decides.
fn windowed_sum(frame: WindowFrame, asc: bool, nulls_first: bool) -> Result<LogicalPlan> {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]);
    let windowed = Expr::from(WindowFunction {
        fun: WindowFunctionDefinition::AggregateUDF(sum_udaf()),
        params: WindowFunctionParams {
            args: vec![col("t.v")],
            partition_by: vec![],
            order_by: vec![datafusion_expr::expr::Sort::new(
                col("t.k"),
                asc,
                nulls_first,
            )],
            window_frame: frame,
            null_treatment: None,
            filter: None,
            distinct: false,
        },
    });
    table_scan(Some("t"), &schema, None)?
        .window(vec![windowed])?
        .build()
}

/// A dialect that accepts only its own NULL placement inside a `RANGE` frame gets
/// the placement as a leading `IS NULL` key, and keeps the rows the plan asked for.
///
/// `GoogleSQL` rejects `NULLS LAST` with `ASC` and `NULLS FIRST` with `DESC` there,
/// and an `ORDER BY` with no explicit frame implies `RANGE` for an aggregate — so
/// the plain `SUM(x) OVER (ORDER BY x)` a plan normalizes to `ASC NULLS LAST` was
/// refused outright.
///
/// Dropping the clause is not the fix and is worse than the failure: `GoogleSQL`
/// defaults to NULLs *first* ascending where `DataFusion` defaults to last, so a
/// dropped clause moves the NULL rows to the other end of the ordering and changes
/// which rows each frame covers. Measured against real BigQuery on
/// `k ∈ {1, 2, NULL, 3}`, `v ∈ {10, 20, 7, 5}`: the placement the plan asks for
/// gives `(NULL,42) (1,10) (2,30) (3,35)`, and dropping it gives
/// `(NULL,7) (1,17) (2,37) (3,42)`.
#[test]
fn test_range_window_nulls_placement_becomes_a_leading_key() -> Result<()> {
    for (asc, nulls_first, direction) in [(true, false, "ASC"), (false, true, "DESC")] {
        let plan = windowed_sum(WindowFrame::new(Some(false)), asc, nulls_first)?;
        let sql = Unparser::new(&BigQueryDialect {})
            .plan_to_sql(&plan)?
            .to_string();

        assert!(
            sql.contains(&format!("`k` IS NULL {direction}")),
            "the NULL placement has to be spelled as a leading key ordered \
             {direction}: {sql}"
        );
        assert!(
            !sql.contains("NULLS LAST") && !sql.contains("NULLS FIRST"),
            "no NULLS clause may survive inside the RANGE frame, which is what \
             BigQuery refuses: {sql}"
        );
        // Non-vacuity: the key itself, its direction and the frame all have to
        // remain, or the statement orders or frames differently.
        assert!(
            sql.contains(&format!(
                "`k` {direction} RANGE BETWEEN UNBOUNDED PRECEDING"
            )) || sql.contains(&format!("`k` {direction} RANGE")),
            "the ordering key and its frame have to survive: {sql}"
        );
    }
    Ok(())
}

/// The placement is rewritten only where the dialect would refuse it. Four
/// controls: its own default placement, a `ROWS` frame, a dialect that accepts any
/// placement, and a `RANGE` frame carrying an offset bound — the last cannot take a
/// second `ORDER BY` key at all ("a RANGE-based window with OFFSET PRECEDING or
/// OFFSET FOLLOWING boundaries must have exactly one ORDER BY key").
#[test]
fn test_range_window_nulls_placement_left_alone_where_it_binds() -> Result<()> {
    let range = WindowFrame::new(Some(false));
    let rows = WindowFrame::new_bounds(
        datafusion_expr::window_frame::WindowFrameUnits::Rows,
        datafusion_expr::window_frame::WindowFrameBound::Preceding(
            datafusion_common::ScalarValue::UInt64(None),
        ),
        datafusion_expr::window_frame::WindowFrameBound::CurrentRow,
    );
    let offset_range = WindowFrame::new_bounds(
        datafusion_expr::window_frame::WindowFrameUnits::Range,
        datafusion_expr::window_frame::WindowFrameBound::Preceding(
            datafusion_common::ScalarValue::UInt64(Some(1)),
        ),
        datafusion_expr::window_frame::WindowFrameBound::CurrentRow,
    );

    // BigQuery's own default placement, a ROWS frame, and an offset RANGE frame.
    for (label, frame, asc, nulls_first) in [
        ("its own default placement", range.clone(), true, true),
        ("a ROWS frame", rows, true, false),
        ("an offset RANGE frame", offset_range, true, false),
    ] {
        let sql = Unparser::new(&BigQueryDialect {})
            .plan_to_sql(&windowed_sum(frame, asc, nulls_first)?)?
            .to_string();
        assert!(
            !sql.contains("IS NULL"),
            "{label}: no leading key is needed here, and adding one to an offset \
             RANGE frame is rejected outright: {sql}"
        );
    }

    // A dialect that accepts any placement keeps the clause it was given.
    let sql = Unparser::new(&UnparserPostgreSqlDialect {})
        .plan_to_sql(&windowed_sum(range, true, false)?)?
        .to_string();
    assert!(
        sql.contains("NULLS LAST") && !sql.contains("IS NULL"),
        "a dialect that can spell the placement must keep spelling it: {sql}"
    );
    Ok(())
}

/// Grouping on a bare column as well as a computed expression: the projection
/// reads the bare column *qualified*, and that qualifier names a relation the new
/// scope encloses. It has to be requalified onto the scope, or the reference binds
/// to nothing.
#[test]
fn test_a_scoped_aggregate_requalifies_a_qualified_group_output() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new(
            "ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
            true,
        ),
    ]);
    let grouped = table_scan(Some("t"), &schema, None)?
        .aggregate(
            vec![
                col("t.a"),
                datafusion_functions::expr_fn::date_trunc(lit("week"), col("t.ts")),
            ],
            vec![count(lit(1i64))],
        )?
        .build()?;
    let outputs = grouped.schema().columns();
    let plan = LogicalPlanBuilder::from(grouped)
        .project(vec![
            Expr::Column(outputs[0].clone()).alias("a"),
            cast(Expr::Column(outputs[1].clone()), DataType::Date32).alias("wk"),
        ])?
        .build()?;

    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    let outer_select = &sql[..sql.find("FROM").expect("a FROM in: {sql}")];
    assert!(
        !outer_select.contains("`t`.`a`"),
        "the outer select still names the enclosed relation, which binds to \
         nothing: {sql}"
    );
    assert!(
        outer_select.contains("`a`"),
        "the grouped column still has to be selected: {sql}"
    );
    Ok(())
}

/// Shapes the scope cannot describe keep the rendering they have today rather than
/// being paired up wrongly or failing to build. Both were found by review and
/// confirmed by rendering them.
#[test]
fn test_the_aggregate_scope_declines_shapes_it_cannot_name() -> Result<()> {
    // A ROLLUP is one `Expr::GroupingSet` covering several outputs, and adds an
    // internal grouping-id output, so the scope's positional pairing does not
    // describe it: it recorded the wrong output as the computed one.
    let rollup_schema = Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new(
            "ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
            true,
        ),
    ]);
    let rollup = table_scan(Some("t"), &rollup_schema, None)?
        .aggregate(
            vec![Expr::GroupingSet(datafusion_expr::GroupingSet::Rollup(
                vec![
                    col("t.a"),
                    datafusion_functions::expr_fn::date_trunc(lit("week"), col("t.ts")),
                ],
            ))],
            vec![count(lit(1i64))],
        )?
        .build()?;
    let outputs = rollup.schema().columns();
    let plan = LogicalPlanBuilder::from(rollup)
        .project(vec![
            cast(Expr::Column(outputs[1].clone()), DataType::Date32).alias("wk"),
        ])?
        .build()?;
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        !sql.contains("derived_aggregate") && sql.contains("ROLLUP"),
        "a grouping set has to keep its flat rendering: {sql}"
    );

    // Grouping a join by two outputs the schema tells apart only by qualifier: the
    // scope names its outputs by the name the schema reports, which drops the
    // qualifier, so naming both would collide and the rewrite would fail to build.
    let left = table_scan(
        Some("l"),
        &Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new(
                "ts",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
                true,
            ),
        ]),
        None,
    )?;
    let right = table_scan(
        Some("r"),
        &Schema::new(vec![Field::new("id", DataType::Int64, true)]),
        None,
    )?
    .build()?;
    let joined = left
        .cross_join(right)?
        .aggregate(
            vec![
                col("l.id"),
                col("r.id"),
                datafusion_functions::expr_fn::date_trunc(lit("week"), col("l.ts")),
            ],
            vec![count(lit(1i64))],
        )?
        .build()?;
    let outputs = joined.schema().columns();
    let plan = LogicalPlanBuilder::from(joined)
        .project(vec![
            cast(Expr::Column(outputs[2].clone()), DataType::Date32).alias("wk"),
        ])?
        .build()?;
    // Before the guard this returned "Schema contains duplicate unqualified field
    // name id" — a query that rendered before stopped rendering at all.
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        !sql.contains("derived_aggregate"),
        "duplicate output names have to keep the flat rendering: {sql}"
    );
    Ok(())
}

/// A volatile window sort key is left alone: the leading key would evaluate it a
/// second time, and could then order by a different value than the key it is
/// placing NULLs within.
#[test]
fn test_a_volatile_range_window_key_keeps_its_nulls_clause() -> Result<()> {
    let plan = {
        let schema = Schema::new(vec![Field::new("v", DataType::Int64, true)]);
        let windowed = Expr::from(WindowFunction {
            fun: WindowFunctionDefinition::AggregateUDF(sum_udaf()),
            params: WindowFunctionParams {
                args: vec![col("t.v")],
                partition_by: vec![],
                order_by: vec![datafusion_expr::expr::Sort::new(
                    datafusion_functions::expr_fn::random(),
                    true,
                    false,
                )],
                window_frame: WindowFrame::new(Some(false)),
                null_treatment: None,
                filter: None,
                distinct: false,
            },
        });
        table_scan(Some("t"), &schema, None)?
            .window(vec![windowed])?
            .build()?
    };
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        !sql.contains("IS NULL"),
        "a volatile key must not be evaluated twice: {sql}"
    );
    Ok(())
}

/// A `HAVING` already on this `SELECT` was classified against the aggregate below
/// and names an expression only the `SELECT` that computes it can name, so the
/// scope is declined rather than stranding it.
///
/// Found by review and confirmed by rendering: before the guard this emitted
/// `… FROM (SELECT … GROUP BY …) AS derived_aggregate_1 HAVING (count(1) > 1)` — a
/// `HAVING` on a `SELECT` that no longer aggregates.
#[test]
fn test_a_having_above_the_projection_declines_the_aggregate_scope() -> Result<()> {
    let schema = Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
        true,
    )]);
    let grouped = table_scan(Some("t"), &schema, None)?
        .aggregate(
            vec![datafusion_functions::expr_fn::date_trunc(
                lit("week"),
                col("t.ts"),
            )],
            vec![count(lit(1i64))],
        )?
        .build()?;
    let outputs = grouped.schema().columns();
    let aggregate_output = outputs[1].name.clone();
    // The aggregate output is passed through unaliased, so the `Filter` above can
    // unproject it into a `HAVING` before this projection is reached.
    let plan = LogicalPlanBuilder::from(grouped)
        .project(vec![
            cast(Expr::Column(outputs[0].clone()), DataType::Date32).alias("wk"),
            Expr::Column(outputs[1].clone()),
        ])?
        .filter(col(aggregate_output).gt(lit(1i64)))?
        .build()?;

    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        sql.contains("HAVING"),
        "the grouped predicate has to survive: {sql}"
    );
    assert!(
        !sql.contains("derived_aggregate"),
        "the aggregate must stay in the SELECT its HAVING names: {sql}"
    );
    Ok(())
}

/// `BigQuery` has no timezone qualifier on a timestamp type: `TIMESTAMP` is the
/// absolute instant and `DATETIME` the civil date-and-time, so the Arrow zone has
/// to select the type rather than decorate it.
///
/// The shape here is the one a dashboard card plans to — a `DATETIME` column
/// compared against a value the source SQL wrote `CAST(x AS TIMESTAMP)` for, which
/// DataFusion types tz-naive. Rendering that cast as `TIMESTAMP` gives the two
/// sides no common supertype and `BigQuery` refuses the whole statement with "No
/// matching signature for operator >= for argument types: DATETIME, TIMESTAMP".
/// Both operands must land on the same type.
///
/// The tz-aware arm is the guard: a zone-carrying cast must keep emitting
/// `TIMESTAMP`, so this cannot be "fixed" by sending every timestamp to
/// `DATETIME` and re-breaking the instant-typed columns.
#[test]
fn test_bigquery_timestamp_cast_follows_the_arrow_timezone() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new(
            "accepted_at",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "decided_at",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
    ]);

    for (cast_type, expected_cast, description) in [
        (
            DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
            "DATETIME",
            "a tz-naive cast is BigQuery's civil type",
        ),
        (
            DataType::Timestamp(
                arrow::datatypes::TimeUnit::Nanosecond,
                Some("UTC".into()),
            ),
            "TIMESTAMP",
            "a zone-carrying cast is BigQuery's instant type",
        ),
    ] {
        let plan = table_scan(Some("ao"), &schema, None)?
            .filter(col("ao.accepted_at").gt_eq(cast(col("ao.decided_at"), cast_type)))?
            .build()?;

        let sql = Unparser::new(&BigQueryDialect {})
            .plan_to_sql(&plan)?
            .to_string();

        assert!(
            sql.contains(&format!("AS {expected_cast})")),
            "{description}: expected a cast to {expected_cast} in: {sql}"
        );
    }

    Ok(())
}

/// A timestamp *literal* is emitted as text that either carries a zone offset or
/// does not, so its cast target has to agree with the string and not only with the
/// Arrow type.
///
/// `BigQuery` reads neither spelling loosely: an offset-bearing string cast to
/// `DATETIME` is rejected ("Invalid datetime string"), and a bare civil string cast
/// to `TIMESTAMP` is silently read as UTC, which is the wrong type to compare
/// against a `DATETIME` column. Every other dialect must keep the rendering it
/// had, which is what makes this a `BigQuery` fix and not a change to the shared
/// literal path.
#[test]
fn test_bigquery_timestamp_literal_cast_follows_the_offset_it_renders() -> Result<()> {
    let naive = lit(datafusion_common::ScalarValue::TimestampNanosecond(
        Some(1_757_934_000_123_456_789),
        None,
    ));
    let aware = lit(datafusion_common::ScalarValue::TimestampNanosecond(
        Some(1_757_934_000_123_456_789),
        Some("+00:00".into()),
    ));

    let bigquery = Unparser::new(&BigQueryDialect {});
    assert!(
        bigquery
            .expr_to_sql(&naive)?
            .to_string()
            .ends_with("AS DATETIME)"),
        "a literal with no offset is the civil type"
    );
    assert!(
        bigquery
            .expr_to_sql(&aware)?
            .to_string()
            .ends_with("AS TIMESTAMP)"),
        "a literal carrying an offset is the instant type"
    );

    // The zone must not reach the cast target in any other dialect.
    let cast_target = |sql: String| sql.rsplit_once("' ").map(|(_, t)| t.to_string());
    for dialect in [
        &UnparserDefaultDialect {} as &dyn UnparserDialect,
        &UnparserPostgreSqlDialect {},
        &UnparserMySqlDialect {},
        &SqliteDialect {},
    ] {
        let unparser = Unparser::new(dialect);
        assert_eq!(
            cast_target(unparser.expr_to_sql(&naive)?.to_string()),
            cast_target(unparser.expr_to_sql(&aware)?.to_string()),
            "the shared literal path must not gain a zone-dependent cast target"
        );
    }

    Ok(())
}

/// DataFusion's internal name for a function is not always the name the target
/// engine answers to. `BigQuery` has no `btrim` and no `now`, and says so —
/// "Function not found: btrim; Did you mean trim?" — which fails the whole
/// statement.
///
/// `btrim` is DataFusion's name for the SQL-standard `TRIM(BOTH ...)`, and
/// `BigQuery`'s `TRIM` trims both ends and takes the optional character set in the
/// same position, so the rename is the entire translation. `now()` carries the
/// session timezone, which is unset by default, so its declared type is the civil
/// one and `CURRENT_DATETIME` is the match — `CURRENT_TIMESTAMP` is an instant,
/// which a `DATETIME` column has no common supertype with.
///
/// The one-sided trims are the guard: `ltrim`/`rtrim` mean something different
/// from `TRIM`, and `BigQuery` matches function names case-insensitively so they
/// already resolve. Renaming them would silently start trimming the other end.
#[test]
fn test_bigquery_renames_only_the_functions_it_cannot_resolve() -> Result<()> {
    use datafusion_functions::datetime::expr_fn as datetime_fn;
    use datafusion_functions::string::expr_fn as string_fn;

    let unparser = Unparser::new(&BigQueryDialect {});

    for (expr, expected, description) in [
        (
            string_fn::btrim(vec![col("s")]),
            "TRIM(`s`)",
            "btrim is BigQuery's TRIM",
        ),
        (
            string_fn::btrim(vec![col("s"), lit("xy")]),
            "TRIM(`s`, 'xy')",
            "the character set keeps its position",
        ),
        (
            datetime_fn::now(),
            "CURRENT_DATETIME()",
            "now() carries the session timezone, unset by default, so its              declared type is the civil one",
        ),
    ] {
        assert_eq!(
            unparser.expr_to_sql(&expr)?.to_string(),
            expected,
            "{description}"
        );
    }

    // The one-sided trims must come through untouched.
    for (expr, name) in [
        (string_fn::ltrim(vec![col("s")]), "ltrim"),
        (string_fn::rtrim(vec![col("s")]), "rtrim"),
    ] {
        let sql = unparser.expr_to_sql(&expr)?.to_string();
        assert!(
            sql.to_lowercase().starts_with(name),
            "{name} trims one end and must not be rewritten to TRIM: {sql}"
        );
    }

    Ok(())
}

/// DataFusion types `date - date` as an `Int64` count of days. BigQuery types
/// `DATE - DATE` as an `INTERVAL`, so the bare operator hands the plan's integer
/// day count to BigQuery as a duration. Whatever the plan does with it next then
/// fails: comparing it to a number gives "No matching signature for operator <
/// for argument types: INTERVAL, INT64", and casting it gives "Invalid cast from
/// INTERVAL to INT64".
///
/// `DATE_DIFF(lhs, rhs, DAY)` is the same quantity with the type the plan means.
///
/// The negative arms carry the weight here. The unparser has no schema, so the
/// rewrite fires only where both operands prove their own date type. A bare
/// column of date type is left alone deliberately: the alternative is guessing,
/// and `date - interval` is a different, legitimate expression that BigQuery
/// accepts and that a date difference would destroy.
#[test]
fn test_bigquery_date_subtraction_becomes_a_day_count() -> Result<()> {
    let unparser = Unparser::new(&BigQueryDialect {});

    let date_difference = cast(col("created_at"), DataType::Date32)
        - cast(col("funding_date"), DataType::Date32);

    assert_eq!(
        unparser.expr_to_sql(&date_difference)?.to_string(),
        "DATE_DIFF(CAST(`created_at` AS DATE), CAST(`funding_date` AS DATE), DAY)",
        "two date operands are a day count"
    );

    // Wrapped in the integer cast the reporting statements apply. The day count is
    // already INT64, so the cast is a no-op rather than the rejected INTERVAL cast.
    assert_eq!(
        unparser
            .expr_to_sql(&cast(date_difference.clone(), DataType::Int32))?
            .to_string(),
        "CAST(DATE_DIFF(CAST(`created_at` AS DATE), CAST(`funding_date` AS DATE), DAY) \
         AS INTEGER)",
        "the integer cast applies to the day count"
    );

    // Negative arms: none of these are a date difference and all must keep `-`.
    let interval = lit(datafusion_common::ScalarValue::IntervalMonthDayNano(Some(
        arrow::datatypes::IntervalMonthDayNano::new(0, 1, 0),
    )));
    for (expr, description) in [
        (
            cast(col("d"), DataType::Date32) - interval,
            "date minus interval is a date, not a day count",
        ),
        (col("a") - col("b"), "bare columns do not prove a date type"),
        (
            cast(col("a"), DataType::Int64) - cast(col("b"), DataType::Int64),
            "numeric operands are ordinary arithmetic",
        ),
        (
            cast(col("a"), DataType::Date32) + cast(col("b"), DataType::Date32),
            "only subtraction is a difference",
        ),
    ] {
        let sql = unparser.expr_to_sql(&expr)?.to_string();
        assert!(!sql.contains("DATE_DIFF"), "{description}: {sql}");
    }

    // Every other dialect keeps the operator.
    assert_eq!(
        Unparser::new(&UnparserDefaultDialect {})
            .expr_to_sql(&date_difference)?
            .to_string(),
        "(CAST(created_at AS DATE) - CAST(funding_date AS DATE))",
        "the rewrite is BigQuery's, not the shared path's"
    );

    Ok(())
}

/// The date rewrites need the operands' types, and expression unparsing has no
/// schema. A plan node does, so the types are made explicit there first.
///
/// These are the two shapes the expression-only rule cannot reach: a subtraction
/// where one side is a bare date column, and an integer cast of one. Both are
/// day counts in the plan; BigQuery refuses the first as `INTERVAL` against
/// `INT64` and the second outright ("Invalid cast from DATE to INT64").
///
/// `UNIX_DATE` is the day number DataFusion means by casting a `Date32` to an
/// integer — zero at the epoch, negative before it — so the two spellings agree
/// on every date, not just the ones a test happens to pick.
#[test]
fn test_bigquery_reads_date_operand_types_from_the_plan() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new(
            "requested_at",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("repayment_date", DataType::Date32, true),
        Field::new("chosen_date", DataType::Date32, true),
        Field::new("decided_on", DataType::Date32, true),
        Field::new("amount", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
    ]);

    // One explicit cast, one bare date column.
    let plan = table_scan(Some("p"), &schema, None)?
        .project(vec![
            (cast(col("p.requested_at"), DataType::Date32) - col("p.repayment_date"))
                .alias("dpd"),
        ])?
        .build()?;
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        sql.contains("DATE_DIFF(") && !sql.contains(" - "),
        "a bare date column still makes a day count: {sql}"
    );

    // Integer casts of two bare date columns.
    let plan = table_scan(Some("r"), &schema, None)?
        .project(vec![
            (cast(col("r.chosen_date"), DataType::Int32)
                - cast(col("r.decided_on"), DataType::Int32))
            .alias("term_days"),
        ])?
        .build()?;
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        sql.matches("UNIX_DATE(").count() == 2,
        "each date-to-integer cast is a day number: {sql}"
    );
    assert!(
        !sql.contains("AS INTEGER"),
        "no cast from DATE to an integer may survive: {sql}"
    );

    // Negative arms: nothing here is a date, and all must keep their cast or
    // operator. A string cast to an integer especially must not become a day
    // number.
    for (projection, description) in [
        (
            cast(col("n.amount"), DataType::Int32).alias("a"),
            "an integer cast of a number",
        ),
        (
            cast(col("n.label"), DataType::Int64).alias("a"),
            "an integer cast of a string",
        ),
        (
            (col("n.amount") - lit(1i64)).alias("a"),
            "numeric subtraction",
        ),
        (
            (col("n.repayment_date")
                - lit(datafusion_common::ScalarValue::IntervalMonthDayNano(Some(
                    arrow::datatypes::IntervalMonthDayNano::new(0, 1, 0),
                ))))
            .alias("a"),
            "a date minus an interval, which is a date",
        ),
    ] {
        let plan = table_scan(Some("n"), &schema, None)?
            .project(vec![projection])?
            .build()?;
        let sql = Unparser::new(&BigQueryDialect {})
            .plan_to_sql(&plan)?
            .to_string();
        assert!(
            !sql.contains("UNIX_DATE") && !sql.contains("DATE_DIFF"),
            "{description} is not a day count: {sql}"
        );
    }

    // The pass is BigQuery's: no other dialect sees a changed plan.
    let plan = table_scan(Some("p"), &schema, None)?
        .project(vec![
            (cast(col("p.requested_at"), DataType::Date32) - col("p.repayment_date"))
                .alias("dpd"),
        ])?
        .build()?;
    assert_eq!(
        Unparser::new(&UnparserDefaultDialect {})
            .plan_to_sql(&plan)?
            .to_string(),
        "SELECT (CAST(p.requested_at AS DATE) - p.repayment_date) AS dpd FROM p",
        "the shared path keeps the operator and adds no cast"
    );

    Ok(())
}

/// `to_unixtime` and `to_timestamp` have no single BigQuery counterpart: the
/// right spelling depends on the operand's type, so a rename cannot do it.
///
/// `UNIX_SECONDS` takes only a `TIMESTAMP` and refuses a `DATETIME` ("No matching
/// signature for function UNIX_SECONDS"), so a civil operand is read as an
/// instant in UTC first — the reading DataFusion gives a tz-naive timestamp and a
/// date. Going the other way, BigQuery has no `TIMESTAMP(int)` ("No matching
/// signature for function TIMESTAMP"); `TIMESTAMP_SECONDS` is the integer form,
/// and its result is an instant, so it is brought back to the civil type
/// `to_timestamp` declares.
///
/// An operand whose type is not evident keeps the old rendering rather than
/// getting a guessed one — the string arms below, where DataFusion's parsing
/// rules are not BigQuery's.
#[test]
fn test_bigquery_unix_time_conversions_follow_the_operand_type() -> Result<()> {
    use datafusion_functions::datetime::expr_fn as datetime_fn;

    let schema = Schema::new(vec![
        Field::new(
            "naive_at",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "aware_at",
            DataType::Timestamp(
                arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            true,
        ),
        Field::new("opened_on", DataType::Date32, true),
        Field::new("secs", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
    ]);

    let rendered = |expr: Expr| -> Result<String> {
        let plan = table_scan(Some("t"), &schema, None)?
            .project(vec![expr.alias("v")])?
            .build()?;
        Ok(Unparser::new(&BigQueryDialect {})
            .plan_to_sql(&plan)?
            .to_string())
    };

    // A civil operand is read as an instant; an instant is used as it is.
    assert!(
        rendered(datetime_fn::to_unixtime(vec![col("t.naive_at")]))?
            .contains("UNIX_SECONDS(TIMESTAMP("),
        "a tz-naive operand has to become an instant first"
    );
    assert!(
        rendered(datetime_fn::to_unixtime(vec![col("t.opened_on")]))?
            .contains("UNIX_SECONDS(TIMESTAMP("),
        "a date operand has to become an instant first"
    );
    let aware = rendered(datetime_fn::to_unixtime(vec![col("t.aware_at")]))?;
    assert!(
        aware.contains("UNIX_SECONDS(") && !aware.contains("UNIX_SECONDS(TIMESTAMP("),
        "an instant operand needs no conversion: {aware}"
    );

    // Integer seconds keep the civil return type the plan declares.
    assert!(
        rendered(datetime_fn::to_timestamp(vec![col("t.secs")]))?
            .contains("DATETIME(TIMESTAMP_SECONDS("),
        "integer seconds are TIMESTAMP_SECONDS, returned as the civil type"
    );

    // Negative arms: a string operand is left exactly as it was.
    for expr in [
        datetime_fn::to_unixtime(vec![col("t.label")]),
        datetime_fn::to_timestamp(vec![col("t.label")]),
    ] {
        let sql = rendered(expr)?;
        assert!(
            !sql.contains("UNIX_SECONDS")
                && !sql.contains("TIMESTAMP_SECONDS")
                && !sql.contains("DATETIME("),
            "a string operand must not get a guessed conversion: {sql}"
        );
    }

    // No other dialect is touched.
    let plan = table_scan(Some("t"), &schema, None)?
        .project(vec![
            datetime_fn::to_unixtime(vec![col("t.naive_at")]).alias("v"),
        ])?
        .build()?;
    assert_eq!(
        Unparser::new(&UnparserDefaultDialect {})
            .plan_to_sql(&plan)?
            .to_string(),
        "SELECT to_unixtime(t.naive_at) AS v FROM t",
        "the shared path keeps DataFusion's own spelling"
    );

    Ok(())
}

/// DataFusion leaves a mismatched comparison in the plan and coerces at
/// execution: it reads a tz-naive timestamp against a tz-aware one as UTC, and a
/// string against a number by reading the string as that number. `BigQuery` has
/// no common supertype for either pair and refuses the whole statement, so the
/// comparison has to be made to agree first.
///
/// The naive side is cast to the other side's zone, which on epoch-based values
/// is a no-op and is the reading DataFusion already gives — confirmed against
/// BigQuery, where `CAST(DATETIME '1970-01-02' AS TIMESTAMP)` equals
/// `TIMESTAMP '1970-01-02+00'`. So the rows do not move; only the SQL says what
/// was already meant.
///
/// The matched-pair arms are the guard: a comparison whose sides already agree
/// must not collect a cast, and no other dialect may see a changed plan.
#[test]
fn test_bigquery_mismatched_comparison_is_made_to_agree() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new(
            "aware",
            DataType::Timestamp(
                arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            true,
        ),
        Field::new(
            "naive",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("num", DataType::Int64, true),
        Field::new("txt", DataType::Utf8, true),
        Field::new("flt", DataType::Float64, true),
    ]);

    let rendered = |expr: Expr, dialect: &dyn UnparserDialect| -> Result<String> {
        let plan = table_scan(Some("t"), &schema, None)?
            .filter(expr)?
            .build()?;
        Ok(Unparser::new(dialect).plan_to_sql(&plan)?.to_string())
    };

    // Either order gets the naive side converted, and only that side.
    for expr in [
        col("t.aware").gt_eq(col("t.naive")),
        col("t.naive").gt_eq(col("t.aware")),
    ] {
        let sql = rendered(expr, &BigQueryDialect {})?;
        assert!(
            sql.contains("CAST(`t`.`naive` AS TIMESTAMP)"),
            "the civil side has to become an instant: {sql}"
        );
        assert!(
            !sql.contains("CAST(`t`.`aware`"),
            "the instant side is already agreeing and must be left alone: {sql}"
        );
    }

    // Sides that already agree collect nothing.
    for expr in [
        col("t.aware").gt_eq(col("t.aware")),
        col("t.naive").gt_eq(col("t.naive")),
    ] {
        let sql = rendered(expr, &BigQueryDialect {})?;
        assert!(
            !sql.contains("CAST("),
            "a matched pair needs no conversion: {sql}"
        );
    }

    // A number against text is the same shape, and the target type is
    // DataFusion's own coercion: it reads the text as the number.
    for expr in [col("t.num").eq(col("t.txt")), col("t.txt").eq(col("t.num"))] {
        let sql = rendered(expr, &BigQueryDialect {})?;
        assert!(
            sql.contains("CAST(`t`.`txt` AS BIGINT)"),
            "the text side has to be read as the number: {sql}"
        );
        assert!(
            !sql.contains("CAST(`t`.`num`"),
            "the numeric side is already the common type: {sql}"
        );
    }

    // A pair the engine coerces on its own is left alone rather than dressed up.
    let coercible = rendered(col("t.num").eq(col("t.flt")), &BigQueryDialect {})?;
    assert!(
        !coercible.contains("CAST("),
        "INT64 against FLOAT64 needs no help: {coercible}"
    );

    // Every other dialect keeps the comparison it had.
    assert_eq!(
        rendered(
            col("t.aware").gt_eq(col("t.naive")),
            &UnparserDefaultDialect {}
        )?,
        "SELECT * FROM t WHERE (t.aware >= t.naive)",
        "only a dialect that refuses the mix rewrites it"
    );

    Ok(())
}

/// BigQuery holds microseconds and its client refuses a longer literal outright
/// rather than rounding it: seven or more sub-second digits come back as "Invalid
/// timestamp: '...'" and fail the statement. A `now()` folded at plan time
/// carries nanoseconds, so a statement whose own SQL holds no literal can still
/// be handed one it will not accept.
///
/// Six digits and fewer pass through untouched, and no other dialect is capped,
/// so a nanosecond literal still reaches an engine that can store it.
#[test]
fn test_bigquery_timestamp_literal_is_truncated_to_microseconds() -> Result<()> {
    let bigquery = Unparser::new(&BigQueryDialect {});

    for (scalar, expected, description) in [
        (
            datafusion_common::ScalarValue::TimestampNanosecond(
                Some(1_776_177_802_860_544_307),
                Some("UTC".into()),
            ),
            "'2026-04-14T14:43:22.860544+00:00'",
            "nanoseconds are cut to microseconds",
        ),
        (
            datafusion_common::ScalarValue::TimestampNanosecond(
                Some(1_776_177_802_860_544_307),
                None,
            ),
            "'2026-04-14 14:43:22.860544'",
            "the civil form is cut the same way",
        ),
        (
            datafusion_common::ScalarValue::TimestampMicrosecond(
                Some(1_776_177_802_860_544),
                Some("UTC".into()),
            ),
            "'2026-04-14T14:43:22.860544+00:00'",
            "a literal already within range is untouched",
        ),
        (
            datafusion_common::ScalarValue::TimestampMillisecond(
                Some(1_776_177_802_860),
                Some("UTC".into()),
            ),
            "'2026-04-14T14:43:22.860+00:00'",
            "fewer digits are not padded",
        ),
    ] {
        let sql = bigquery.expr_to_sql(&lit(scalar))?.to_string();
        assert!(sql.contains(expected), "{description}: {sql}");
    }

    // No other dialect is capped: a nanosecond literal survives in full.
    let nanos = lit(datafusion_common::ScalarValue::TimestampNanosecond(
        Some(1_776_177_802_860_544_307),
        Some("UTC".into()),
    ));
    for dialect in [
        &UnparserDefaultDialect {} as &dyn UnparserDialect,
        &UnparserPostgreSqlDialect {},
    ] {
        let sql = Unparser::new(dialect).expr_to_sql(&nanos)?.to_string();
        assert!(
            sql.contains(".860544307"),
            "an uncapped dialect keeps every digit: {sql}"
        );
    }

    Ok(())
}

/// A constant grouping key partitions nothing — every row carries the same value
/// — so leaving it out groups identically and removes an emission engines
/// disagree about. `BigQuery` refuses it outright ("Cannot GROUP BY literal
/// values"), and an engine that reads a bare integer in `GROUP BY` as a
/// select-list ordinal reads it as something other than the constant the plan
/// meant.
///
/// The constant still projects; only the grouping loses it. An all-constant
/// grouping leaves no keys at all, which is a single group over the input and the
/// same result again.
#[test]
fn test_constant_group_by_keys_are_left_out() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("g", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]);

    let grouped = |keys: Vec<Expr>| -> Result<LogicalPlan> {
        table_scan(Some("t"), &schema, None)?
            .aggregate(keys, vec![count(lit(1i64))])?
            .build()
    };

    // A constant beside a real key: the key survives, the constant does not.
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&grouped(vec![lit("POOLED"), col("t.g")])?)?
        .to_string();
    assert!(
        sql.contains("GROUP BY `t`.`g`"),
        "the real key has to survive: {sql}"
    );
    let group_by_clause = &sql[sql.find("GROUP BY").expect("a GROUP BY")..];
    assert!(
        !group_by_clause.contains("'POOLED'"),
        "the constant must not reach GROUP BY: {sql}"
    );
    assert!(
        sql.contains("'POOLED'"),
        "the constant still belongs in the select list: {sql}"
    );

    // Every dialect gains this, since grouping by a constant means the same
    // everywhere and an ordinal reading would mean something else.
    for dialect in [
        &UnparserDefaultDialect {} as &dyn UnparserDialect,
        &UnparserPostgreSqlDialect {},
    ] {
        let sql = Unparser::new(dialect)
            .plan_to_sql(&grouped(vec![lit(1i64), col("t.g")])?)?
            .to_string();
        assert!(
            !sql.contains("GROUP BY 1"),
            "a bare integer key would be read as an ordinal: {sql}"
        );
    }

    Ok(())
}

/// `BigQuery` has no aggregate percentile. `PERCENTILE_CONT` exists only as an
/// analytic function and says so — "percentile_cont aggregate function is not
/// supported" — and `APPROX_QUANTILES`, which is an aggregate, answers a
/// different question: over `[1,2,3,4]` it returns 2 where the median is 2.5.
///
/// Both `median` and `approx_percentile_cont` are therefore rendered by ordering
/// the group into an array and interpolating between the two values the
/// percentile falls between. Checked against BigQuery on the emitted statement:
/// the medians of `[1,2,3,4]` and `[10,20,31]` come back 2.5 and 20.0, and the
/// p95s 3.85 and 29.9 — matching `PERCENTILE_CONT` to floating-point noise, and
/// matching DataFusion's `median`, which averages the two middle values.
#[test]
fn test_bigquery_percentile_aggregates_are_rendered_by_ordering_the_group() -> Result<()>
{
    let schema = Schema::new(vec![
        Field::new("g", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
    ]);

    let rendered = |agg: Expr, dialect: &dyn UnparserDialect| -> Result<String> {
        let plan = table_scan(Some("t"), &schema, None)?
            .aggregate(vec![col("t.g")], vec![agg])?
            .build()?;
        Ok(Unparser::new(dialect).plan_to_sql(&plan)?.to_string())
    };

    let median = || datafusion_functions_aggregate::median::median(col("t.v"));

    let sql = rendered(median(), &BigQueryDialect {})?;
    assert!(
        !sql.contains("median("),
        "BigQuery has no median to call: {sql}"
    );
    assert!(
        sql.contains("ARRAY_AGG(`t`.`v` IGNORE NULLS ORDER BY `t`.`v`)"),
        "the group has to be ordered, and a null must not take a position: {sql}"
    );
    assert!(
        sql.contains("FLOOR(") && sql.contains("CEIL("),
        "the two straddling values are what the result interpolates between: {sql}"
    );
    assert!(
        !sql.contains("APPROX_QUANTILES"),
        "an approximation would answer a different question: {sql}"
    );

    // Every other dialect keeps its own aggregate.
    assert!(
        rendered(median(), &UnparserDefaultDialect {})?.contains("median("),
        "only BigQuery lacks the aggregate"
    );

    Ok(())
}

/// An expression unparsed for one node may have been unprojected from a node
/// below it — a select item substituted back into the aggregate expression that
/// produced it — and it then references that node's input rather than this one's.
///
/// Those nodes flatten into the same `SELECT`, so their inputs are in scope for
/// the expressions that end up in it. Without that, a type-directed rendering
/// declines on the way back up: `to_unixtime` over a column of the aggregate's
/// input stayed untranslated and reached BigQuery as `Function not found:
/// to_unixtime`, even though the same expression translates when the column's own
/// schema is in hand.
#[test]
fn test_type_directed_rendering_survives_unprojection() -> Result<()> {
    use datafusion_functions::datetime::expr_fn as datetime_fn;

    let schema = Schema::new(vec![
        Field::new(
            "canceled_at",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "first_rej_at",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
    ]);

    let days = cast(
        (datetime_fn::to_unixtime(vec![col("c.canceled_at")])
            - datetime_fn::to_unixtime(vec![col("c.first_rej_at")]))
            / lit(86400i64),
        DataType::Int64,
    );
    let grouped = table_scan(Some("c"), &schema, None)?
        .aggregate(vec![days.alias("days_to_cancel")], vec![count(lit(1i64))])?
        .build()?;
    let outputs = grouped.schema().columns();
    let plan = LogicalPlanBuilder::from(grouped)
        .project(vec![
            Expr::Column(outputs[0].clone()),
            Expr::Column(outputs[1].clone()),
        ])?
        .build()?;

    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        !sql.contains("to_unixtime"),
        "the unprojected expression still has to be translated: {sql}"
    );
    assert!(
        sql.contains("UNIX_SECONDS(TIMESTAMP(`c`.`canceled_at`))"),
        "and translated by the operand's real type: {sql}"
    );

    Ok(())
}

/// A grouping key appears in both an aggregate's output and its input, so walking
/// down for the scope an unprojected expression resolves in meets the same name
/// twice. Treating that as an ambiguity and stopping loses the rest of the scope,
/// and a rendering that needs a type then declines: a tz-aware column compared
/// against a tz-naive expression stayed as it was and reached BigQuery as "No
/// matching signature for operator < for argument types: TIMESTAMP, DATETIME".
///
/// The repeat is the same column flowing through, so it is skipped. A repeat
/// carrying a different type under one name is a real ambiguity and still stops
/// the walk, since guessing which one a reference means is worse than declining.
#[test]
fn test_a_grouping_key_repeated_in_scope_does_not_hide_the_rest() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new(
            "deposit_ts",
            DataType::Timestamp(
                arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            true,
        ),
        Field::new("cohort_start", DataType::Date32, true),
        Field::new("tok", DataType::Utf8, true),
    ]);

    let week = lit(datafusion_common::ScalarValue::IntervalMonthDayNano(Some(
        arrow::datatypes::IntervalMonthDayNano::new(0, 7, 0),
    )));
    let window_end = cast(
        col("c.cohort_start"),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
    ) + week;

    // The reported shape: the comparison sits inside an aggregate, and the
    // grouping key is `cohort_start`, which the comparison also reads.
    let counted = datafusion_functions_aggregate::expr_fn::count_distinct(
        datafusion_expr::when(col("c.deposit_ts").lt(window_end), col("c.tok"))
            .otherwise(lit(datafusion_common::ScalarValue::Null))?,
    );
    let grouped = table_scan(Some("c"), &schema, None)?
        .aggregate(vec![col("c.cohort_start")], vec![counted])?
        .build()?;
    let outputs = grouped.schema().columns();
    let plan = LogicalPlanBuilder::from(grouped)
        .project(vec![
            Expr::Column(outputs[0].clone()),
            Expr::Column(outputs[1].clone()),
        ])?
        .build()?;

    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        sql.contains("AS TIMESTAMP)"),
        "the civil side of the comparison has to be made an instant: {sql}"
    );

    Ok(())
}

/// The renderings have to reach every shape that means the same thing, or a
/// statement fails on the one that was missed.
///
/// `TRY_CAST(date AS INT64)` is the plain cast with a different keyword, and
/// BigQuery refuses `SAFE_CAST(DATE AS INT64)` exactly as it refuses `CAST`.
/// `BETWEEN` compares the way `<=` does, so a dialect that will not mix the
/// operand types refuses it for the same reason — confirmed against BigQuery,
/// where a `DATETIME BETWEEN TIMESTAMP AND TIMESTAMP` is rejected and the same
/// comparison with the civil side converted is accepted.
#[test]
fn test_bigquery_renderings_reach_try_cast_and_between() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("d", DataType::Date32, true),
        Field::new(
            "naive",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "aware",
            DataType::Timestamp(
                arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            true,
        ),
    ]);

    let rendered = |expr: Expr| -> Result<String> {
        let plan = table_scan(Some("t"), &schema, None)?
            .filter(expr)?
            .build()?;
        Ok(Unparser::new(&BigQueryDialect {})
            .plan_to_sql(&plan)?
            .to_string())
    };

    // A try-cast of a date to an integer is the same day count.
    let day_number = Expr::TryCast(datafusion_expr::expr::TryCast::new(
        Box::new(col("t.d")),
        DataType::Int64,
    ));
    let sql = rendered(day_number.gt(lit(0i64)))?;
    assert!(
        sql.contains("UNIX_DATE("),
        "a try-cast is refused the same way, so it needs the same rendering: {sql}"
    );
    assert!(
        !sql.contains("AS BIGINT") && !sql.contains("AS INT64"),
        "no integer cast of a date may survive: {sql}"
    );

    // BETWEEN's operands have to agree, and only the civil one moves.
    let sql = rendered(Expr::Between(datafusion_expr::expr::Between::new(
        Box::new(col("t.naive")),
        false,
        Box::new(col("t.aware")),
        Box::new(col("t.aware")),
    )))?;
    assert!(
        sql.contains("CAST(`t`.`naive` AS TIMESTAMP)"),
        "the civil side has to become an instant: {sql}"
    );
    assert!(
        !sql.contains("CAST(`t`.`aware`"),
        "the instant sides already agree: {sql}"
    );

    Ok(())
}

/// DataFusion's `median` declares the decimal type it was given and coerces only
/// integers to float. A median needs no interpolation — at an integer index both
/// offsets address the same element — so it is rendered as the average of the two
/// straddling values, which keeps the input's type. Confirmed against BigQuery: a
/// `NUMERIC` column returns `NUMERIC` with its precision intact, and an `INT64`
/// column returns a float, as DataFusion does.
#[test]
fn test_bigquery_median_keeps_the_input_type() -> Result<()> {
    let schema = Schema::new(vec![Field::new("v", DataType::Decimal128(20, 4), true)]);
    let plan = table_scan(Some("t"), &schema, None)?
        .aggregate(
            Vec::<Expr>::new(),
            vec![datafusion_functions_aggregate::median::median(col("t.v"))],
        )?
        .build()?;
    let sql = Unparser::new(&BigQueryDialect {})
        .plan_to_sql(&plan)?
        .to_string();
    assert!(
        !sql.contains("FLOAT64"),
        "a decimal median must not be pushed through a float: {sql}"
    );
    assert!(
        sql.contains(") / 2"),
        "the median is the average of the two straddling values: {sql}"
    );

    Ok(())
}
