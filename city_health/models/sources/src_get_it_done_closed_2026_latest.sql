with raw as (
    select *
    from read_csv_auto('data/raw/latest/get_it_done_closed_2026_latest.csv', header = true, all_varchar = true)
)

select
    service_request_id,
    service_request_parent_id,
    sap_notification_number,
    date_requested,
    case_age_days,
    case_record_type,
    service_name,
    service_name_detail,
    date_closed,
    status,
    lat,
    lng,
    street_address,
    zipcode,
    council_district,
    comm_plan_code,
    comm_plan_name,
    park_name,
    case_origin,
    referred,
    iamfloc,
    floc,
    public_description,
    null::varchar as specify_the_issue,
    'closed_2026'::varchar as source_dataset
from raw
