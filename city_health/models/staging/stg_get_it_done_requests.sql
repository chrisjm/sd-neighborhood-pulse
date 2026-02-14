with unioned as (
    select * from {{ ref('src_get_it_done_open_current_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2026_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2025_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2024_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2023_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2022_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2021_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2020_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2019_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2018_latest') }}
    union all
    select * from {{ ref('src_get_it_done_closed_2017_latest') }}
)

select
    trim(service_request_id) as service_request_id,
    nullif(trim(service_request_parent_id), '') as service_request_parent_id,
    nullif(trim(sap_notification_number), '') as sap_notification_number,
    try_cast(date_requested as timestamp) as date_requested,
    try_cast(case_age_days as integer) as case_age_days,
    nullif(trim(case_record_type), '') as case_record_type,
    nullif(trim(service_name), '') as service_name,
    nullif(trim(service_name_detail), '') as service_name_detail,
    try_cast(date_closed as timestamp) as date_closed,
    nullif(trim(status), '') as status,
    case
        when lower(trim(status)) in ('open', 'closed', 'referred') then lower(trim(status))
        else 'other'
    end as status_bucket,
    try_cast(lat as double) as latitude,
    try_cast(lng as double) as longitude,
    nullif(trim(street_address), '') as street_address,
    nullif(trim(zipcode), '') as zipcode,
    nullif(trim(council_district), '') as council_district,
    nullif(trim(comm_plan_code), '') as comm_plan_code,
    nullif(trim(comm_plan_name), '') as comm_plan_name,
    nullif(trim(park_name), '') as park_name,
    nullif(trim(case_origin), '') as case_origin,
    nullif(trim(referred), '') as referred,
    nullif(trim(iamfloc), '') as iamfloc,
    nullif(trim(floc), '') as floc,
    nullif(trim(public_description), '') as public_description,
    nullif(trim(specify_the_issue), '') as specify_the_issue,
    source_dataset
from unioned
