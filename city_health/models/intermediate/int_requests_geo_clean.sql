select
    service_request_id,
    source_dataset,
    date_requested,
    date_closed,
    cast(date_requested as date) as requested_date,
    cast(date_closed as date) as closed_date,
    status,
    status_bucket,
    case_age_days,
    service_name,
    service_name_detail,
    case_record_type,
    case_origin,
    service_request_parent_id,
    referred,
    public_description,
    latitude,
    longitude,
    coalesce(nullif(trim(comm_plan_name), ''), 'Unknown') as comm_plan_name,
    coalesce(nullif(trim(council_district), ''), 'Unknown') as council_district,
    coalesce(nullif(trim(zipcode), ''), 'Unknown') as zipcode,
    coalesce(cast(case_age_days as integer), datediff('day', cast(date_requested as date), current_date)) as derived_case_age_days,
    case when status_bucket = 'open' then 1 else 0 end as is_open,
    case when status_bucket in ('closed', 'referred') then 1 else 0 end as is_closed_or_referred
from {{ ref('stg_get_it_done_requests') }}
where latitude is not null
  and longitude is not null
  and latitude between -90 and 90
  and longitude between -180 and 180
  and date_requested is not null
