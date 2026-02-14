{% snapshot snp_get_it_done_requests %}
{{
    config(
        unique_key='request_snapshot_key',
        strategy='check',
        check_cols=[
            'status_bucket',
            'status',
            'date_closed',
            'case_age_days',
            'service_name',
            'service_name_detail',
            'latitude',
            'longitude',
            'comm_plan_name',
            'council_district',
            'zipcode',
            'referred'
        ],
        invalidate_hard_deletes=True
    )
}}

select
    concat(coalesce(service_request_id, ''), '||', coalesce(source_dataset, '')) as request_snapshot_key,
    *
from {{ ref('stg_get_it_done_requests') }}
{% endsnapshot %}
