select *
from {{ ref('fct_neighborhood_frustration_index') }}
where frustration_index < 0
   or frustration_index > 100
   or backlog_component < 0
   or backlog_component > 100
   or aging_component < 0
   or aging_component > 100
   or repeat_component < 0
   or repeat_component > 100
   or resolution_component < 0
   or resolution_component > 100
