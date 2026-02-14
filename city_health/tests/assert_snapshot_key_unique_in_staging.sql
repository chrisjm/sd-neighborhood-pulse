select
    concat(coalesce(service_request_id, ''), '||', coalesce(source_dataset, '')) as request_snapshot_key,
    count(*) as row_count
from {{ ref('stg_get_it_done_requests') }}
group by 1
having count(*) > 1
