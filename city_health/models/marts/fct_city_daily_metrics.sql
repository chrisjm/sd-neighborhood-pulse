with requested_daily as (
    select
        requested_date as metric_date,
        count(*) as request_count,
        sum(is_open) as open_request_count,
        sum(case when is_open = 1 and derived_case_age_days > 14 then 1 else 0 end) as aging_open_request_count,
        sum(case when service_request_parent_id is not null and trim(service_request_parent_id) <> '' then 1 else 0 end) as duplicate_child_request_count,
        median(resolution_days) as median_resolution_days
    from {{ ref('int_requests_enriched_time') }}
    group by 1
),
closed_daily as (
    select
        closed_date as metric_date,
        count(*) as closed_request_count
    from {{ ref('int_requests_enriched_time') }}
    where closed_date is not null
    group by 1
),
combined as (
    select
        coalesce(r.metric_date, c.metric_date) as metric_date,
        coalesce(r.request_count, 0) as request_count,
        coalesce(r.open_request_count, 0) as open_request_count,
        coalesce(r.aging_open_request_count, 0) as aging_open_request_count,
        coalesce(r.duplicate_child_request_count, 0) as duplicate_child_request_count,
        coalesce(r.median_resolution_days, 0) as median_resolution_days,
        coalesce(c.closed_request_count, 0) as closed_request_count
    from requested_daily r
    full outer join closed_daily c
      on r.metric_date = c.metric_date
),
rolling as (
    select
        *,
        sum(request_count) over (
            order by metric_date
            range between interval '2 days' preceding and current row
        ) as opened_request_count_3d,
        sum(request_count) over (
            order by metric_date
            range between interval '6 days' preceding and current row
        ) as opened_request_count_7d,
        sum(closed_request_count) over (
            order by metric_date
            range between interval '2 days' preceding and current row
        ) as closed_request_count_3d,
        sum(closed_request_count) over (
            order by metric_date
            range between interval '6 days' preceding and current row
        ) as closed_request_count_7d
    from combined
)
select
    metric_date,
    request_count,
    open_request_count,
    aging_open_request_count,
    duplicate_child_request_count,
    median_resolution_days,
    closed_request_count,
    opened_request_count_3d,
    opened_request_count_7d,
    closed_request_count_3d,
    closed_request_count_7d
from rolling
