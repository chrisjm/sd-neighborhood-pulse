select *
from {{ ref('int_requests_geo_clean') }}
where zipcode <> 'Unknown'
  and regexp_full_match(zipcode, '^[0-9]{5}$') = false
