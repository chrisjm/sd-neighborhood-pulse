with windows as (
    select 30 as window_days
    union all
    select 90 as window_days
),
filtered as (
    select
        w.window_days,
        r.comm_plan_name,
        r.council_district,
        r.zipcode,
        r.service_name,
        r.status_bucket,
        r.latitude,
        r.longitude,
        floor(r.latitude * 200) / 200.0 as lat_bin,
        floor(r.longitude * 200) / 200.0 as lon_bin
    from windows w
    join {{ ref('int_requests_enriched_time') }} r
      on r.requested_date >= current_date - (w.window_days - 1)
),
cluster_rollup as (
    select
        window_days,
        comm_plan_name,
        council_district,
        zipcode,
        lat_bin,
        lon_bin,
        avg(latitude) as centroid_latitude,
        avg(longitude) as centroid_longitude,
        count(*) as request_count,
        sum(case when status_bucket = 'open' then 1 else 0 end) as open_request_count,
        sum(case when status_bucket in ('closed', 'referred') then 1 else 0 end) as closed_or_referred_count
    from filtered
    group by 1, 2, 3, 4, 5, 6
),
service_counts as (
    select
        window_days,
        comm_plan_name,
        council_district,
        zipcode,
        lat_bin,
        lon_bin,
        service_name,
        count(*) as service_request_count
    from filtered
    group by 1, 2, 3, 4, 5, 6, 7
),
ranked_service as (
    select
        *,
        row_number() over (
            partition by window_days, comm_plan_name, council_district, zipcode, lat_bin, lon_bin
            order by service_request_count desc, service_name
        ) as service_rank
    from service_counts
)
select
    concat(
        'wd',
        cast(r.window_days as varchar),
        '::',
        coalesce(r.comm_plan_name, 'Unknown'),
        '::',
        coalesce(r.council_district, 'Unknown'),
        '::',
        coalesce(r.zipcode, 'Unknown'),
        '::',
        cast(r.lat_bin as varchar),
        '::',
        cast(r.lon_bin as varchar)
    ) as cluster_id,
    r.window_days,
    r.comm_plan_name,
    r.council_district,
    r.zipcode,
    r.lat_bin,
    r.lon_bin,
    r.centroid_latitude,
    r.centroid_longitude,
    r.request_count,
    r.open_request_count,
    r.closed_or_referred_count,
    case
        when r.request_count = 0 then 0
        else round((r.open_request_count * 1.0 / r.request_count) * 100, 2)
    end as open_ratio_pct,
    s.service_name as dominant_service_name,
    coalesce(s.service_request_count, 0) as dominant_service_request_count,
    current_date as as_of_date
from cluster_rollup r
left join ranked_service s
  on r.window_days = s.window_days
 and r.comm_plan_name = s.comm_plan_name
 and r.council_district = s.council_district
 and r.zipcode = s.zipcode
 and r.lat_bin = s.lat_bin
 and r.lon_bin = s.lon_bin
 and s.service_rank = 1
where r.request_count >= 5
