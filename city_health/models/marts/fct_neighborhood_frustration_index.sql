with raw_metrics as (
    select * from {{ ref('fct_neighborhood_daily_metrics') }}
),
comm_plan_lookup as (
    select * from {{ ref('int_comm_plan_geo_lookup') }}
),
grains as (
    select distinct grain_type, grain_value
    from raw_metrics
),
date_bounds as (
    select
        min(metric_date) as min_date,
        max(metric_date) as max_date
    from raw_metrics
),
date_spine as (
    select cast(series_date as date) as metric_date
    from date_bounds,
    generate_series(min_date, max_date, interval 1 day) as t(series_date)
),
filled_metrics as (
    select
        d.metric_date,
        g.grain_type,
        g.grain_value,
        coalesce(m.request_count, 0) as request_count,
        coalesce(m.open_request_count, 0) as open_request_count,
        coalesce(m.aging_open_request_count, 0) as aging_open_request_count,
        coalesce(m.duplicate_child_request_count, 0) as duplicate_child_request_count,
        coalesce(m.median_resolution_days, 0) as median_resolution_days
    from date_spine d
    cross join grains g
    left join raw_metrics m
      on d.metric_date = m.metric_date
     and g.grain_type = m.grain_type
     and g.grain_value = m.grain_value
),
rolling as (
    select
        metric_date,
        grain_type,
        grain_value,
        sum(request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 29 preceding and current row
        ) as request_count_30d,
        sum(open_request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 29 preceding and current row
        ) as open_request_count_30d,
        sum(aging_open_request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 29 preceding and current row
        ) as aging_open_request_count_30d,
        sum(duplicate_child_request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 29 preceding and current row
        ) as duplicate_child_request_count_30d,
        avg(median_resolution_days) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 29 preceding and current row
        ) as avg_resolution_days_30d,
        sum(request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 89 preceding and current row
        ) as request_count_90d,
        sum(open_request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 89 preceding and current row
        ) as open_request_count_90d,
        sum(aging_open_request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 89 preceding and current row
        ) as aging_open_request_count_90d,
        sum(duplicate_child_request_count) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 89 preceding and current row
        ) as duplicate_child_request_count_90d,
        avg(median_resolution_days) over (
            partition by grain_type, grain_value
            order by metric_date
            rows between 89 preceding and current row
        ) as avg_resolution_days_90d
    from filled_metrics
),
windowed as (
    select
        metric_date as as_of_date,
        grain_type,
        grain_value,
        30 as window_days,
        request_count_30d as request_count,
        open_request_count_30d as open_request_count,
        aging_open_request_count_30d as aging_open_request_count,
        duplicate_child_request_count_30d as duplicate_child_request_count,
        avg_resolution_days_30d as avg_resolution_days
    from rolling

    union all

    select
        metric_date as as_of_date,
        grain_type,
        grain_value,
        90 as window_days,
        request_count_90d as request_count,
        open_request_count_90d as open_request_count,
        aging_open_request_count_90d as aging_open_request_count,
        duplicate_child_request_count_90d as duplicate_child_request_count,
        avg_resolution_days_90d as avg_resolution_days
    from rolling
),
component_scores as (
    select
        as_of_date,
        grain_type,
        grain_value,
        window_days,
        request_count,
        open_request_count,
        aging_open_request_count,
        duplicate_child_request_count,
        avg_resolution_days,
        case when request_count = 0 then 0 else (open_request_count * 1.0 / request_count) * 100 end as backlog_component,
        case when open_request_count = 0 then 0 else (aging_open_request_count * 1.0 / open_request_count) * 100 end as aging_component,
        case when request_count = 0 then 0 else (duplicate_child_request_count * 1.0 / request_count) * 100 end as repeat_component,
        least(100.0, coalesce(avg_resolution_days, 0) / 30.0 * 100.0) as resolution_component
    from windowed
),
component_scores_mapped as (
    select
        c.*,
        case
            when c.grain_type = 'comm_plan_name'
                then coalesce(l.comm_plan_geo_name, c.grain_value)
            else c.grain_value
        end as grain_geo_value
    from component_scores c
    left join comm_plan_lookup l
      on c.grain_type = 'comm_plan_name'
     and case
            when trim(
                regexp_replace(
                    regexp_replace(lower(replace(c.grain_value, '&', ' and ')), '[^a-z0-9 ]+', ' ', 'g'),
                    '\\s+',
                    ' ',
                    'g'
                )
            ) like 'reserve area%'
                then 'reserve'
            else trim(
                regexp_replace(
                    regexp_replace(lower(replace(c.grain_value, '&', ' and ')), '[^a-z0-9 ]+', ' ', 'g'),
                    '\\s+',
                    ' ',
                    'g'
                )
            )
        end = l.comm_plan_normalized
)
select
    as_of_date,
    grain_type,
    grain_value,
    grain_geo_value,
    case when grain_type = 'comm_plan_name' then true else false end as is_primary_grain,
    window_days,
    request_count,
    open_request_count,
    aging_open_request_count,
    duplicate_child_request_count,
    round(avg_resolution_days, 2) as avg_resolution_days,
    round(backlog_component, 2) as backlog_component,
    round(aging_component, 2) as aging_component,
    round(repeat_component, 2) as repeat_component,
    round(resolution_component, 2) as resolution_component,
    round((backlog_component + aging_component + repeat_component + resolution_component) / 4.0, 2) as frustration_index
from component_scores_mapped
