select
    service_request_id,
    source_dataset,
    count(*) as row_count
from {{ ref('stg_get_it_done_requests') }}
group by 1, 2
having count(*) > 1
