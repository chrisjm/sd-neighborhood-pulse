select
    cluster_id,
    window_days,
    count(*) as row_count
from {{ ref('fct_request_hotspots') }}
group by 1, 2
having count(*) > 1
