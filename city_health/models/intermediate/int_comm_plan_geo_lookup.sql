with geojson_raw as (
    select features
    from read_json_auto('data/geojson/comm_plan_codes.geojson')
),
flattened_features as (
    select unnest(features) as feature
    from geojson_raw
),
extracted as (
    select
        cast(feature.properties.CPCODE as varchar) as comm_plan_code,
        cast(feature.properties.CPNAME as varchar) as comm_plan_geo_name
    from flattened_features
),
normalized as (
    select
        comm_plan_code,
        comm_plan_geo_name,
        trim(
            regexp_replace(
                regexp_replace(lower(replace(comm_plan_geo_name, '&', ' and ')), '[^a-z0-9 ]+', ' ', 'g'),
                '\\s+',
                ' ',
                'g'
            )
        ) as comm_plan_normalized
    from extracted
)
select distinct
    case
        when comm_plan_normalized like 'reserve area%'
            then 'reserve'
        else comm_plan_normalized
    end as comm_plan_normalized,
    comm_plan_code,
    comm_plan_geo_name
from normalized
where comm_plan_geo_name is not null
  and trim(comm_plan_geo_name) <> ''
