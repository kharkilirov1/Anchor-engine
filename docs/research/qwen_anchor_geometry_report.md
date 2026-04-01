# Qwen Anchor Geometry Report

## Summary

- Model: `Qwen/Qwen2.5-1.5B`
- Layer indices analyzed: `0..27` (28 total model layers; embedding state kept only for reference)
- Clean cases: `12`
- Noisy cases: `0`
- Skipped cases: `0`
- Full-span clean verdict: `partial_separation`
- Trimmed-span clean verdict: `no_separation`
- Support after tokenization controls: `not_supported`
- Cases retaining full geometry after trimming: `12` / `12`

## Tokenization audit table

| case | class | group | audit | match | token_count | leading_ws | same_count | same_ids | trimmed_full_geometry | decoded_tokens | issues |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| content_vegan_brief | content_like | strictly_vegan_meal_plan_policy | clean | offset_mapping | 5 | True | True | True | True | ` strictly |  vegan |  meal |  plan |  policy` | - |
| content_vegan_reason | content_like | strictly_vegan_meal_plan_policy | clean | offset_mapping | 5 | True | True | True | True | ` strictly |  vegan |  meal |  plan |  policy` | - |
| content_fastapi_architecture | content_like | async_fastapi_service_architecture_policy | clean | offset_mapping | 6 | True | True | True | True | ` async |  Fast | API |  service |  architecture |  policy` | - |
| content_fastapi_summary | content_like | async_fastapi_service_architecture_policy | clean | offset_mapping | 6 | True | True | True | True | ` async |  Fast | API |  service |  architecture |  policy` | - |
| content_json_contract | content_like | json_only_response_format_policy | clean | offset_mapping | 5 | True | True | True | True | ` JSON |  only |  response |  format |  policy` | - |
| content_json_parser | content_like | json_only_response_format_policy | clean | offset_mapping | 5 | True | True | True | True | ` JSON |  only |  response |  format |  policy` | - |
| procedure_contradiction_proof | procedure_like | proof_by_contradiction_reasoning_steps | clean | offset_mapping | 5 | True | True | True | True | ` proof |  by |  contradiction |  reasoning |  steps` | - |
| procedure_contradiction_explain | procedure_like | proof_by_contradiction_reasoning_steps | clean | offset_mapping | 5 | True | True | True | True | ` proof |  by |  contradiction |  reasoning |  steps` | - |
| procedure_binary_search_note | procedure_like | binary_search_update_loop_procedure | clean | offset_mapping | 5 | True | True | True | True | ` binary |  search |  update |  loop |  procedure` | - |
| procedure_binary_search_indices | procedure_like | binary_search_update_loop_procedure | clean | offset_mapping | 5 | True | True | True | True | ` binary |  search |  update |  loop |  procedure` | - |
| procedure_di_request_path | procedure_like | dependency_injection_request_flow_sequence | clean | offset_mapping | 5 | True | True | True | True | ` dependency |  injection |  request |  flow |  sequence` | - |
| procedure_di_summary | procedure_like | dependency_injection_request_flow_sequence | clean | offset_mapping | 5 | True | True | True | True | ` dependency |  injection |  request |  flow |  sequence` | - |

## Class-level layer curves — full span (clean only)

| layer | content coherence | procedure coherence | gap | content tortuosity | procedure tortuosity | gap | content rank1 EV | procedure rank1 EV | gap | content stability | procedure stability | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L00 | -0.467 | -0.464 | -0.003 | 4.060 | 3.839 | 0.220 | 0.431 | 0.454 | -0.023 | 0.963 | 0.972 | -0.009 |
| L01 | -0.462 | -0.473 | 0.011 | 4.174 | 3.963 | 0.211 | 0.472 | 0.485 | -0.013 | 0.944 | 0.954 | -0.010 |
| L02 | -0.438 | -0.445 | 0.007 | 3.879 | 3.623 | 0.256 | 0.451 | 0.463 | -0.012 | 0.968 | 0.965 | 0.003 |
| L03 | -0.452 | -0.442 | -0.010 | 4.001 | 3.538 | 0.464 | 0.462 | 0.446 | 0.016 | 0.951 | 0.944 | 0.008 |
| L04 | -0.449 | -0.428 | -0.022 | 3.923 | 3.492 | 0.432 | 0.457 | 0.466 | -0.009 | 0.933 | 0.924 | 0.009 |
| L05 | -0.448 | -0.416 | -0.033 | 3.916 | 3.591 | 0.325 | 0.452 | 0.465 | -0.013 | 0.923 | 0.902 | 0.021 |
| L06 | -0.452 | -0.412 | -0.040 | 3.893 | 3.533 | 0.360 | 0.456 | 0.482 | -0.025 | 0.909 | 0.888 | 0.022 |
| L07 | -0.434 | -0.405 | -0.030 | 3.780 | 3.470 | 0.310 | 0.472 | 0.484 | -0.012 | 0.910 | 0.884 | 0.026 |
| L08 | -0.436 | -0.419 | -0.017 | 3.762 | 3.583 | 0.179 | 0.481 | 0.521 | -0.039 | 0.903 | 0.881 | 0.022 |
| L09 | -0.443 | -0.438 | -0.005 | 3.763 | 3.668 | 0.095 | 0.480 | 0.546 | -0.066 | 0.893 | 0.872 | 0.022 |
| L10 | -0.431 | -0.433 | 0.003 | 3.646 | 3.591 | 0.055 | 0.476 | 0.541 | -0.065 | 0.893 | 0.877 | 0.016 |
| L11 | -0.411 | -0.413 | 0.001 | 3.601 | 3.475 | 0.126 | 0.458 | 0.519 | -0.061 | 0.884 | 0.880 | 0.004 |
| L12 | -0.412 | -0.404 | -0.008 | 3.585 | 3.426 | 0.159 | 0.458 | 0.511 | -0.053 | 0.876 | 0.872 | 0.004 |
| L13 | -0.400 | -0.392 | -0.009 | 3.554 | 3.342 | 0.212 | 0.445 | 0.492 | -0.048 | 0.871 | 0.868 | 0.003 |
| L14 | -0.403 | -0.420 | 0.017 | 3.482 | 3.468 | 0.014 | 0.455 | 0.523 | -0.068 | 0.870 | 0.855 | 0.015 |
| L15 | -0.397 | -0.410 | 0.013 | 3.479 | 3.408 | 0.070 | 0.458 | 0.512 | -0.053 | 0.871 | 0.861 | 0.010 |
| L16 | -0.390 | -0.397 | 0.007 | 3.470 | 3.368 | 0.102 | 0.452 | 0.498 | -0.046 | 0.866 | 0.858 | 0.008 |
| L17 | -0.386 | -0.390 | 0.004 | 3.432 | 3.323 | 0.109 | 0.462 | 0.498 | -0.037 | 0.855 | 0.846 | 0.009 |
| L18 | -0.370 | -0.378 | 0.008 | 3.393 | 3.224 | 0.169 | 0.454 | 0.492 | -0.038 | 0.829 | 0.836 | -0.007 |
| L19 | -0.375 | -0.391 | 0.015 | 3.358 | 3.255 | 0.104 | 0.461 | 0.502 | -0.041 | 0.834 | 0.829 | 0.005 |
| L20 | -0.368 | -0.392 | 0.023 | 3.372 | 3.215 | 0.156 | 0.471 | 0.504 | -0.033 | 0.819 | 0.826 | -0.007 |
| L21 | -0.364 | -0.386 | 0.022 | 3.374 | 3.185 | 0.189 | 0.466 | 0.507 | -0.041 | 0.823 | 0.811 | 0.012 |
| L22 | -0.348 | -0.378 | 0.031 | 3.376 | 3.173 | 0.203 | 0.481 | 0.518 | -0.037 | 0.808 | 0.799 | 0.009 |
| L23 | -0.336 | -0.363 | 0.027 | 3.383 | 3.141 | 0.242 | 0.500 | 0.533 | -0.033 | 0.774 | 0.780 | -0.006 |
| L24 | -0.316 | -0.356 | 0.039 | 3.349 | 3.208 | 0.141 | 0.496 | 0.555 | -0.059 | 0.737 | 0.742 | -0.005 |
| L25 | -0.303 | -0.350 | 0.048 | 3.327 | 3.228 | 0.099 | 0.484 | 0.564 | -0.080 | 0.706 | 0.708 | -0.002 |
| L26 | -0.305 | -0.351 | 0.047 | 3.324 | 3.216 | 0.108 | 0.486 | 0.570 | -0.085 | 0.687 | 0.690 | -0.003 |
| L27 | -0.244 | -0.276 | 0.031 | 3.171 | 2.953 | 0.218 | 0.570 | 0.587 | -0.017 | 0.653 | 0.683 | -0.030 |

## Class-level layer curves — trimmed span (clean only)

| layer | content coherence | procedure coherence | gap | content tortuosity | procedure tortuosity | gap | content rank1 EV | procedure rank1 EV | gap | content stability | procedure stability | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L00 | -0.452 | -0.443 | -0.010 | 3.060 | 2.789 | 0.271 | 0.506 | 0.535 | -0.028 | 0.987 | 0.989 | -0.002 |
| L01 | -0.464 | -0.425 | -0.039 | 3.267 | 2.664 | 0.603 | 0.554 | 0.541 | 0.013 | 0.977 | 0.978 | -0.001 |
| L02 | -0.416 | -0.434 | 0.017 | 2.880 | 2.732 | 0.148 | 0.524 | 0.556 | -0.032 | 0.981 | 0.980 | 0.000 |
| L03 | -0.420 | -0.438 | 0.018 | 2.878 | 2.742 | 0.136 | 0.514 | 0.546 | -0.032 | 0.963 | 0.967 | -0.005 |
| L04 | -0.427 | -0.422 | -0.005 | 2.969 | 2.654 | 0.315 | 0.510 | 0.558 | -0.048 | 0.941 | 0.956 | -0.015 |
| L05 | -0.421 | -0.413 | -0.008 | 2.885 | 2.599 | 0.286 | 0.505 | 0.562 | -0.057 | 0.934 | 0.956 | -0.022 |
| L06 | -0.431 | -0.385 | -0.047 | 2.940 | 2.487 | 0.453 | 0.512 | 0.564 | -0.052 | 0.928 | 0.954 | -0.026 |
| L07 | -0.401 | -0.371 | -0.030 | 2.825 | 2.452 | 0.373 | 0.517 | 0.570 | -0.053 | 0.928 | 0.954 | -0.026 |
| L08 | -0.398 | -0.381 | -0.017 | 2.819 | 2.459 | 0.360 | 0.519 | 0.605 | -0.086 | 0.921 | 0.955 | -0.034 |
| L09 | -0.415 | -0.390 | -0.025 | 2.869 | 2.414 | 0.455 | 0.515 | 0.617 | -0.101 | 0.907 | 0.951 | -0.044 |
| L10 | -0.403 | -0.388 | -0.014 | 2.837 | 2.420 | 0.416 | 0.515 | 0.615 | -0.100 | 0.906 | 0.950 | -0.044 |
| L11 | -0.382 | -0.374 | -0.008 | 2.746 | 2.396 | 0.350 | 0.509 | 0.601 | -0.092 | 0.899 | 0.944 | -0.044 |
| L12 | -0.384 | -0.365 | -0.019 | 2.723 | 2.401 | 0.322 | 0.514 | 0.592 | -0.078 | 0.883 | 0.936 | -0.053 |
| L13 | -0.376 | -0.359 | -0.017 | 2.716 | 2.409 | 0.307 | 0.508 | 0.574 | -0.066 | 0.873 | 0.928 | -0.056 |
| L14 | -0.374 | -0.382 | 0.008 | 2.706 | 2.413 | 0.293 | 0.514 | 0.597 | -0.083 | 0.863 | 0.927 | -0.064 |
| L15 | -0.366 | -0.371 | 0.005 | 2.662 | 2.395 | 0.267 | 0.513 | 0.585 | -0.072 | 0.869 | 0.923 | -0.053 |
| L16 | -0.357 | -0.354 | -0.003 | 2.642 | 2.368 | 0.274 | 0.508 | 0.568 | -0.060 | 0.861 | 0.908 | -0.047 |
| L17 | -0.351 | -0.345 | -0.006 | 2.614 | 2.358 | 0.256 | 0.506 | 0.568 | -0.062 | 0.847 | 0.904 | -0.057 |
| L18 | -0.332 | -0.331 | -0.001 | 2.574 | 2.328 | 0.246 | 0.500 | 0.557 | -0.057 | 0.830 | 0.886 | -0.056 |
| L19 | -0.347 | -0.360 | 0.013 | 2.610 | 2.448 | 0.161 | 0.514 | 0.579 | -0.066 | 0.824 | 0.879 | -0.055 |
| L20 | -0.332 | -0.362 | 0.030 | 2.544 | 2.470 | 0.074 | 0.512 | 0.575 | -0.063 | 0.815 | 0.864 | -0.049 |
| L21 | -0.324 | -0.358 | 0.034 | 2.535 | 2.456 | 0.079 | 0.504 | 0.576 | -0.072 | 0.813 | 0.847 | -0.034 |
| L22 | -0.306 | -0.339 | 0.034 | 2.475 | 2.379 | 0.096 | 0.510 | 0.587 | -0.077 | 0.799 | 0.838 | -0.039 |
| L23 | -0.287 | -0.317 | 0.030 | 2.414 | 2.292 | 0.122 | 0.530 | 0.598 | -0.068 | 0.773 | 0.827 | -0.054 |
| L24 | -0.263 | -0.297 | 0.035 | 2.348 | 2.211 | 0.137 | 0.523 | 0.621 | -0.098 | 0.736 | 0.813 | -0.077 |
| L25 | -0.252 | -0.283 | 0.030 | 2.360 | 2.183 | 0.178 | 0.514 | 0.622 | -0.108 | 0.688 | 0.789 | -0.101 |
| L26 | -0.249 | -0.276 | 0.027 | 2.340 | 2.176 | 0.163 | 0.510 | 0.626 | -0.116 | 0.673 | 0.773 | -0.099 |
| L27 | -0.169 | -0.175 | 0.005 | 2.052 | 1.959 | 0.093 | 0.569 | 0.661 | -0.091 | 0.687 | 0.799 | -0.112 |

## Group-level layer curves — full span (clean only)

| group | class | transitional | strongest content layer | strongest procedure layer | margin curve |
| --- | --- | --- | ---: | ---: | --- |
| async_fastapi_service_architecture_policy | content_like | no | 2 | 18 | L00:0.688 L01:0.607 L02:0.697 L03:0.666 L04:0.654 L05:0.630 L06:0.604 L07:0.585 L08:0.574 L09:0.565 L10:0.549 L11:0.532 L12:0.529 L13:0.492 L14:0.490 L15:0.492 L16:0.483 L17:0.474 L18:0.433 L19:0.505 L20:0.512 L21:0.526 L22:0.532 L23:0.513 L24:0.499 L25:0.493 L26:0.487 L27:0.459 |
| binary_search_update_loop_procedure | procedure_like | no | 18 | 2 | L00:-0.483 L01:-0.438 L02:-0.512 L03:-0.490 L04:-0.472 L05:-0.442 L06:-0.432 L07:-0.414 L08:-0.392 L09:-0.391 L10:-0.367 L11:-0.353 L12:-0.336 L13:-0.333 L14:-0.336 L15:-0.325 L16:-0.327 L17:-0.306 L18:-0.273 L19:-0.351 L20:-0.338 L21:-0.318 L22:-0.352 L23:-0.378 L24:-0.380 L25:-0.395 L26:-0.373 L27:-0.331 |
| dependency_injection_request_flow_sequence | procedure_like | no | 18 | 0 | L00:-0.615 L01:-0.592 L02:-0.597 L03:-0.577 L04:-0.602 L05:-0.593 L06:-0.582 L07:-0.563 L08:-0.564 L09:-0.563 L10:-0.554 L11:-0.541 L12:-0.546 L13:-0.513 L14:-0.530 L15:-0.526 L16:-0.530 L17:-0.539 L18:-0.493 L19:-0.523 L20:-0.530 L21:-0.522 L22:-0.551 L23:-0.530 L24:-0.525 L25:-0.526 L26:-0.499 L27:-0.550 |
| json_only_response_format_policy | content_like | no | 0 | 18 | L00:0.650 L01:0.592 L02:0.624 L03:0.608 L04:0.592 L05:0.576 L06:0.557 L07:0.537 L08:0.543 L09:0.542 L10:0.533 L11:0.517 L12:0.509 L13:0.480 L14:0.495 L15:0.482 L16:0.482 L17:0.474 L18:0.430 L19:0.489 L20:0.491 L21:0.502 L22:0.491 L23:0.479 L24:0.480 L25:0.502 L26:0.485 L27:0.549 |
| proof_by_contradiction_reasoning_steps | procedure_like | no | 27 | 0 | L00:-0.531 L01:-0.504 L02:-0.502 L03:-0.520 L04:-0.512 L05:-0.492 L06:-0.496 L07:-0.454 L08:-0.477 L09:-0.488 L10:-0.490 L11:-0.472 L12:-0.480 L13:-0.450 L14:-0.463 L15:-0.458 L16:-0.464 L17:-0.468 L18:-0.443 L19:-0.441 L20:-0.444 L21:-0.498 L22:-0.463 L23:-0.438 L24:-0.441 L25:-0.455 L26:-0.468 L27:-0.408 |
| strictly_vegan_meal_plan_policy | content_like | no | 2 | 27 | L00:0.679 L01:0.634 L02:0.713 L03:0.671 L04:0.653 L05:0.631 L06:0.599 L07:0.571 L08:0.570 L09:0.573 L10:0.551 L11:0.546 L12:0.551 L13:0.515 L14:0.546 L15:0.539 L16:0.556 L17:0.555 L18:0.496 L19:0.535 L20:0.535 L21:0.564 L22:0.577 L23:0.571 L24:0.567 L25:0.580 L26:0.539 L27:0.269 |

## Group-level layer curves — trimmed span (clean only)

| group | class | transitional | strongest content layer | strongest procedure layer | margin curve |
| --- | --- | --- | ---: | ---: | --- |
| async_fastapi_service_architecture_policy | content_like | no | 2 | 18 | L00:0.626 L01:0.592 L02:0.699 L03:0.669 L04:0.639 L05:0.629 L06:0.623 L07:0.605 L08:0.605 L09:0.605 L10:0.604 L11:0.569 L12:0.572 L13:0.550 L14:0.562 L15:0.549 L16:0.531 L17:0.513 L18:0.473 L19:0.553 L20:0.546 L21:0.555 L22:0.571 L23:0.578 L24:0.562 L25:0.549 L26:0.547 L27:0.502 |
| binary_search_update_loop_procedure | procedure_like | no | 26 | 2 | L00:-0.504 L01:-0.463 L02:-0.513 L03:-0.505 L04:-0.489 L05:-0.500 L06:-0.479 L07:-0.454 L08:-0.467 L09:-0.493 L10:-0.506 L11:-0.451 L12:-0.435 L13:-0.421 L14:-0.449 L15:-0.421 L16:-0.407 L17:-0.398 L18:-0.390 L19:-0.426 L20:-0.396 L21:-0.372 L22:-0.364 L23:-0.364 L24:-0.328 L25:-0.326 L26:-0.309 L27:-0.376 |
| dependency_injection_request_flow_sequence | procedure_like | no | 27 | 3 | L00:-0.504 L01:-0.562 L02:-0.593 L03:-0.597 L04:-0.567 L05:-0.519 L06:-0.570 L07:-0.555 L08:-0.590 L09:-0.580 L10:-0.592 L11:-0.552 L12:-0.556 L13:-0.522 L14:-0.573 L15:-0.548 L16:-0.540 L17:-0.533 L18:-0.490 L19:-0.541 L20:-0.531 L21:-0.510 L22:-0.506 L23:-0.499 L24:-0.456 L25:-0.446 L26:-0.446 L27:-0.406 |
| json_only_response_format_policy | content_like | no | 10 | 27 | L00:0.613 L01:0.585 L02:0.644 L03:0.651 L04:0.645 L05:0.621 L06:0.632 L07:0.618 L08:0.647 L09:0.661 L10:0.675 L11:0.628 L12:0.630 L13:0.604 L14:0.639 L15:0.627 L16:0.617 L17:0.610 L18:0.542 L19:0.563 L20:0.555 L21:0.542 L22:0.516 L23:0.487 L24:0.504 L25:0.516 L26:0.490 L27:0.206 |
| proof_by_contradiction_reasoning_steps | procedure_like | no | 27 | 24 | L00:-0.501 L01:-0.402 L02:-0.438 L03:-0.452 L04:-0.481 L05:-0.508 L06:-0.519 L07:-0.495 L08:-0.471 L09:-0.456 L10:-0.482 L11:-0.469 L12:-0.462 L13:-0.474 L14:-0.432 L15:-0.466 L16:-0.473 L17:-0.455 L18:-0.421 L19:-0.439 L20:-0.474 L21:-0.498 L22:-0.502 L23:-0.512 L24:-0.572 L25:-0.571 L26:-0.558 L27:-0.338 |
| strictly_vegan_meal_plan_policy | content_like | no | 0 | 27 | L00:0.641 L01:0.593 L02:0.621 L03:0.602 L04:0.581 L05:0.580 L06:0.566 L07:0.542 L08:0.515 L09:0.508 L10:0.529 L11:0.486 L12:0.456 L13:0.472 L14:0.493 L15:0.478 L16:0.483 L17:0.471 L18:0.454 L19:0.486 L20:0.498 L21:0.500 L22:0.490 L23:0.472 L24:0.488 L25:0.498 L26:0.486 L27:0.244 |

## Full-span vs trimmed-span comparison

| subset | verdict | case_count | max separation layer | first positive layer | stable birth layer |
| --- | --- | ---: | ---: | ---: | ---: |
| full_span / all_valid | partial_separation | 12 | 22 | n/a | n/a |
| full_span / clean_only | partial_separation | 12 | 22 | n/a | n/a |
| trimmed_span / all_valid | no_separation | 12 | 2 | n/a | n/a |
| trimmed_span / clean_only | no_separation | 12 | 2 | n/a | n/a |

## Layer of maximal separation

- Full span / clean only: `{'layer': 22, 'separation_score': 0.5, 'positive_signals': 2, 'available_signals': 4}`
- Trimmed span / clean only: `{'layer': 2, 'separation_score': 0.5, 'positive_signals': 2, 'available_signals': 4}`

## Whether evidence supports polarity-from-geometry

Current judgment: `not_supported`.

The strongest evidence should come from clean-only trimmed spans, because that setting removes the first-token boundary artifact while keeping the original tokenizer intact. If full-span and trimmed-span disagree, trimmed-span is treated as the stricter control.

## Limitations

- The prompt set is still small and local, so group-level conclusions can be noisy.
- All cases retain full geometry after trimming, so the trimmed-span result is not explained by metric collapse from short spans.
- Cross-prompt stability is defined within the paired prompts already present in the probe, not over a large paraphrase set.
- Mean-direction affinity is a diagnostic proxy and not a causal proof of polarity.
- Some groups may remain transitional because the phrase itself mixes content and procedure semantics.

## Recommended next step

Run a tightly paired paraphrase probe per anchor group, holding token count fixed while varying only the surrounding sentence frame.
