select *
from {{ ref('int_requests_geo_clean') }}
where council_district <> 'Unknown'
  and regexp_full_match(council_district, '^District [0-9]{1,2}$') = false
