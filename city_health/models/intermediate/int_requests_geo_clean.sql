with source_values as (
    select
        service_request_id,
        source_dataset,
        date_requested,
        date_closed,
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
        nullif(regexp_replace(trim(comm_plan_name), '\\s+', ' '), '') as comm_plan_name_clean,
        nullif(regexp_replace(trim(council_district), '\\s+', ' '), '') as council_district_clean,
        nullif(regexp_replace(trim(zipcode), '\\s+', ''), '') as zipcode_clean
    from {{ ref('stg_get_it_done_requests') }}
),
normalized_values as (
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
        case
            when comm_plan_name_clean is null then 'Unknown'
            else array_to_string(
                list_transform(
                    string_split(lower(comm_plan_name_clean), ' '),
                    token -> upper(left(token, 1)) || substr(token, 2)
                ),
                ' '
            )
        end as comm_plan_name,
        case
            when council_district_clean is null then 'Unknown'
            when regexp_extract(council_district_clean, '([0-9]{1,2})', 1) <> ''
                then concat('District ', regexp_extract(council_district_clean, '([0-9]{1,2})', 1))
            else 'Unknown'
        end as council_district,
        case
            when regexp_extract(coalesce(zipcode_clean, ''), '([0-9]{5})', 1) <> ''
                then regexp_extract(zipcode_clean, '([0-9]{5})', 1)
            else 'Unknown'
        end as zipcode,
        coalesce(cast(case_age_days as integer), datediff('day', cast(date_requested as date), current_date)) as derived_case_age_days,
        case when status_bucket = 'open' then 1 else 0 end as is_open,
        case when status_bucket in ('closed', 'referred') then 1 else 0 end as is_closed_or_referred
    from source_values
)
select *
from normalized_values
where latitude is not null
  and longitude is not null
  and latitude between -90 and 90
  and longitude between -180 and 180
  and date_requested is not null
