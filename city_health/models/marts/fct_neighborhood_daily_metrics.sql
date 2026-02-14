with grain_expanded as (
    select
        requested_date as metric_date,
        'comm_plan_name' as grain_type,
        comm_plan_name as grain_value,
        is_open,
        derived_case_age_days,
        resolution_days,
        service_request_parent_id
    from {{ ref('int_requests_enriched_time') }}

    union all

    select
        requested_date as metric_date,
        'council_district' as grain_type,
        council_district as grain_value,
        is_open,
        derived_case_age_days,
        resolution_days,
        service_request_parent_id
    from {{ ref('int_requests_enriched_time') }}

    union all

    select
        requested_date as metric_date,
        'zipcode' as grain_type,
        zipcode as grain_value,
        is_open,
        derived_case_age_days,
        resolution_days,
        service_request_parent_id
    from {{ ref('int_requests_enriched_time') }}
)
select
    metric_date,
    grain_type,
    grain_value,
    count(*) as request_count,
    sum(is_open) as open_request_count,
    sum(case when is_open = 1 and derived_case_age_days > 14 then 1 else 0 end) as aging_open_request_count,
    sum(case when service_request_parent_id is not null and trim(service_request_parent_id) <> '' then 1 else 0 end) as duplicate_child_request_count,
    median(resolution_days) as median_resolution_days
from grain_expanded
group by 1, 2, 3
