select
    *,
    date_trunc('week', requested_date) as requested_week,
    date_trunc('month', requested_date) as requested_month,
    case
        when is_closed_or_referred = 1 and closed_date is not null
            then greatest(datediff('day', requested_date, closed_date), 0)
        else null
    end as resolution_days
from {{ ref('int_requests_geo_clean') }}
