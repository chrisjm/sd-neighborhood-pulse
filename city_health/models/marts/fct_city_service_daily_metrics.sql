with daily as (
    select
        requested_date as metric_date,
        coalesce(nullif(trim(service_name), ''), 'Unknown') as service_name,
        count(*) as request_count
    from {{ ref('int_requests_enriched_time') }}
    group by 1, 2
),
rolling_bounds as (
    select
        metric_date,
        service_name,
        request_count,
        quantile_cont(request_count, 0.05) over (
            partition by service_name
            order by metric_date
            rows between 89 preceding and 1 preceding
        ) as p05_90d,
        quantile_cont(request_count, 0.95) over (
            partition by service_name
            order by metric_date
            rows between 89 preceding and 1 preceding
        ) as p95_90d
    from daily
),
scored as (
    select
        metric_date,
        service_name,
        request_count,
        p05_90d,
        p95_90d,
        case
            when p95_90d is not null and request_count >= p95_90d then true
            else false
        end as is_busy_day,
        case
            when p05_90d is not null and request_count <= p05_90d then true
            else false
        end as is_light_day
    from rolling_bounds
)
select *
from scored
