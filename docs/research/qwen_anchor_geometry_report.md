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
- Cases retaining full geometry after trimming: `2` / `12`

## Tokenization audit table

| case | class | group | audit | match | token_count | leading_ws | same_count | same_ids | trimmed_full_geometry | decoded_tokens | issues |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| content_vegan_brief | content_like | strictly_vegan_meal_plan | clean | offset_mapping | 4 | True | True | True | False | ` strictly |  vegan |  meal |  plan` | - |
| content_vegan_reason | content_like | strictly_vegan_meal_plan | clean | offset_mapping | 4 | True | True | True | False | ` strictly |  vegan |  meal |  plan` | - |
| content_fastapi_architecture | content_like | async_fastapi_service_design | clean | offset_mapping | 5 | True | True | True | True | ` async |  Fast | API |  service |  design` | - |
| content_fastapi_summary | content_like | async_fastapi_service_design | clean | offset_mapping | 5 | True | True | True | True | ` async |  Fast | API |  service |  design` | - |
| content_json_contract | content_like | json_only_response_format | clean | offset_mapping | 4 | True | True | True | False | ` JSON |  only |  response |  format` | - |
| content_json_parser | content_like | json_only_response_format | clean | offset_mapping | 4 | True | True | True | False | ` JSON |  only |  response |  format` | - |
| procedure_contradiction_proof | procedure_like | proof_by_contradiction_method | clean | offset_mapping | 4 | True | True | True | False | ` proof |  by |  contradiction |  method` | - |
| procedure_contradiction_explain | procedure_like | proof_by_contradiction_method | clean | offset_mapping | 4 | True | True | True | False | ` proof |  by |  contradiction |  method` | - |
| procedure_binary_search_note | procedure_like | binary_search_update_loop | clean | offset_mapping | 4 | True | True | True | False | ` binary |  search |  update |  loop` | - |
| procedure_binary_search_indices | procedure_like | binary_search_update_loop | clean | offset_mapping | 4 | True | True | True | False | ` binary |  search |  update |  loop` | - |
| procedure_di_request_path | procedure_like | dependency_injection_request_flow | clean | offset_mapping | 4 | True | True | True | False | ` dependency |  injection |  request |  flow` | - |
| procedure_di_summary | procedure_like | dependency_injection_request_flow | clean | offset_mapping | 4 | True | True | True | False | ` dependency |  injection |  request |  flow` | - |

## Class-level layer curves — full span (clean only)

| layer | content coherence | procedure coherence | gap | content tortuosity | procedure tortuosity | gap | content rank1 EV | procedure rank1 EV | gap | content stability | procedure stability | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L00 | -0.429 | -0.488 | 0.059 | 2.990 | 2.906 | 0.084 | 0.526 | 0.572 | -0.047 | 0.969 | 0.975 | -0.005 |
| L01 | -0.411 | -0.482 | 0.071 | 3.037 | 2.911 | 0.126 | 0.562 | 0.593 | -0.031 | 0.954 | 0.960 | -0.006 |
| L02 | -0.446 | -0.492 | 0.045 | 3.283 | 2.931 | 0.352 | 0.565 | 0.587 | -0.021 | 0.963 | 0.964 | -0.001 |
| L03 | -0.453 | -0.473 | 0.020 | 3.268 | 2.785 | 0.483 | 0.574 | 0.569 | 0.004 | 0.948 | 0.944 | 0.004 |
| L04 | -0.445 | -0.473 | 0.029 | 3.150 | 2.820 | 0.330 | 0.569 | 0.586 | -0.017 | 0.935 | 0.919 | 0.016 |
| L05 | -0.445 | -0.457 | 0.012 | 3.116 | 2.879 | 0.237 | 0.561 | 0.576 | -0.014 | 0.926 | 0.899 | 0.027 |
| L06 | -0.447 | -0.469 | 0.022 | 3.132 | 2.925 | 0.206 | 0.561 | 0.588 | -0.027 | 0.915 | 0.880 | 0.035 |
| L07 | -0.446 | -0.473 | 0.027 | 3.089 | 2.875 | 0.214 | 0.569 | 0.585 | -0.016 | 0.912 | 0.879 | 0.033 |
| L08 | -0.457 | -0.497 | 0.040 | 3.081 | 2.999 | 0.082 | 0.577 | 0.615 | -0.038 | 0.909 | 0.874 | 0.034 |
| L09 | -0.456 | -0.517 | 0.062 | 3.029 | 3.061 | -0.032 | 0.578 | 0.641 | -0.063 | 0.900 | 0.860 | 0.041 |
| L10 | -0.446 | -0.515 | 0.070 | 2.999 | 2.981 | 0.018 | 0.570 | 0.634 | -0.064 | 0.898 | 0.871 | 0.026 |
| L11 | -0.435 | -0.495 | 0.060 | 2.972 | 2.926 | 0.046 | 0.551 | 0.616 | -0.066 | 0.887 | 0.872 | 0.015 |
| L12 | -0.434 | -0.485 | 0.051 | 2.960 | 2.850 | 0.110 | 0.549 | 0.610 | -0.061 | 0.883 | 0.861 | 0.022 |
| L13 | -0.421 | -0.464 | 0.043 | 2.915 | 2.728 | 0.187 | 0.535 | 0.593 | -0.058 | 0.870 | 0.852 | 0.019 |
| L14 | -0.430 | -0.497 | 0.067 | 2.903 | 2.837 | 0.066 | 0.545 | 0.621 | -0.076 | 0.864 | 0.837 | 0.027 |
| L15 | -0.422 | -0.485 | 0.063 | 2.897 | 2.775 | 0.123 | 0.545 | 0.610 | -0.066 | 0.867 | 0.846 | 0.021 |
| L16 | -0.415 | -0.471 | 0.055 | 2.881 | 2.753 | 0.128 | 0.539 | 0.600 | -0.062 | 0.861 | 0.841 | 0.021 |
| L17 | -0.410 | -0.464 | 0.054 | 2.855 | 2.711 | 0.144 | 0.546 | 0.597 | -0.051 | 0.850 | 0.834 | 0.016 |
| L18 | -0.393 | -0.447 | 0.054 | 2.785 | 2.619 | 0.166 | 0.537 | 0.588 | -0.051 | 0.828 | 0.821 | 0.006 |
| L19 | -0.401 | -0.471 | 0.070 | 2.733 | 2.633 | 0.100 | 0.542 | 0.596 | -0.054 | 0.833 | 0.822 | 0.011 |
| L20 | -0.391 | -0.463 | 0.072 | 2.739 | 2.577 | 0.162 | 0.551 | 0.600 | -0.049 | 0.821 | 0.823 | -0.002 |
| L21 | -0.390 | -0.449 | 0.059 | 2.745 | 2.583 | 0.162 | 0.546 | 0.599 | -0.053 | 0.815 | 0.800 | 0.015 |
| L22 | -0.374 | -0.444 | 0.070 | 2.736 | 2.612 | 0.125 | 0.553 | 0.605 | -0.052 | 0.809 | 0.778 | 0.031 |
| L23 | -0.373 | -0.436 | 0.063 | 2.763 | 2.630 | 0.133 | 0.568 | 0.613 | -0.045 | 0.774 | 0.756 | 0.018 |
| L24 | -0.355 | -0.426 | 0.071 | 2.750 | 2.710 | 0.040 | 0.565 | 0.625 | -0.060 | 0.740 | 0.723 | 0.017 |
| L25 | -0.341 | -0.425 | 0.083 | 2.710 | 2.733 | -0.023 | 0.552 | 0.633 | -0.081 | 0.710 | 0.693 | 0.017 |
| L26 | -0.333 | -0.429 | 0.096 | 2.693 | 2.738 | -0.045 | 0.552 | 0.639 | -0.087 | 0.695 | 0.678 | 0.017 |
| L27 | -0.277 | -0.343 | 0.066 | 2.609 | 2.557 | 0.053 | 0.619 | 0.635 | -0.016 | 0.696 | 0.642 | 0.054 |

## Class-level layer curves — trimmed span (clean only)

| layer | content coherence | procedure coherence | gap | content tortuosity | procedure tortuosity | gap | content rank1 EV | procedure rank1 EV | gap | content stability | procedure stability | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L00 | -0.422 | n/a | n/a | 2.703 | n/a | n/a | 0.556 | n/a | n/a | 0.988 | 0.990 | -0.002 |
| L01 | -0.454 | n/a | n/a | 3.163 | n/a | n/a | 0.680 | n/a | n/a | 0.980 | 0.981 | -0.001 |
| L02 | -0.444 | n/a | n/a | 2.827 | n/a | n/a | 0.645 | n/a | n/a | 0.981 | 0.979 | 0.002 |
| L03 | -0.432 | n/a | n/a | 2.789 | n/a | n/a | 0.621 | n/a | n/a | 0.961 | 0.968 | -0.006 |
| L04 | -0.442 | n/a | n/a | 2.853 | n/a | n/a | 0.636 | n/a | n/a | 0.944 | 0.954 | -0.010 |
| L05 | -0.418 | n/a | n/a | 2.722 | n/a | n/a | 0.604 | n/a | n/a | 0.938 | 0.956 | -0.018 |
| L06 | -0.424 | n/a | n/a | 2.774 | n/a | n/a | 0.620 | n/a | n/a | 0.933 | 0.956 | -0.022 |
| L07 | -0.388 | n/a | n/a | 2.661 | n/a | n/a | 0.598 | n/a | n/a | 0.934 | 0.952 | -0.017 |
| L08 | -0.387 | n/a | n/a | 2.616 | n/a | n/a | 0.584 | n/a | n/a | 0.929 | 0.949 | -0.019 |
| L09 | -0.392 | n/a | n/a | 2.681 | n/a | n/a | 0.602 | n/a | n/a | 0.913 | 0.943 | -0.030 |
| L10 | -0.393 | n/a | n/a | 2.648 | n/a | n/a | 0.590 | n/a | n/a | 0.914 | 0.945 | -0.031 |
| L11 | -0.379 | n/a | n/a | 2.550 | n/a | n/a | 0.552 | n/a | n/a | 0.905 | 0.938 | -0.034 |
| L12 | -0.375 | n/a | n/a | 2.534 | n/a | n/a | 0.554 | n/a | n/a | 0.888 | 0.930 | -0.041 |
| L13 | -0.377 | n/a | n/a | 2.520 | n/a | n/a | 0.548 | n/a | n/a | 0.871 | 0.917 | -0.046 |
| L14 | -0.383 | n/a | n/a | 2.553 | n/a | n/a | 0.560 | n/a | n/a | 0.860 | 0.910 | -0.050 |
| L15 | -0.368 | n/a | n/a | 2.532 | n/a | n/a | 0.553 | n/a | n/a | 0.859 | 0.906 | -0.047 |
| L16 | -0.369 | n/a | n/a | 2.489 | n/a | n/a | 0.549 | n/a | n/a | 0.853 | 0.893 | -0.040 |
| L17 | -0.359 | n/a | n/a | 2.472 | n/a | n/a | 0.540 | n/a | n/a | 0.838 | 0.884 | -0.045 |
| L18 | -0.341 | n/a | n/a | 2.396 | n/a | n/a | 0.528 | n/a | n/a | 0.817 | 0.858 | -0.041 |
| L19 | -0.331 | n/a | n/a | 2.301 | n/a | n/a | 0.528 | n/a | n/a | 0.810 | 0.840 | -0.030 |
| L20 | -0.290 | n/a | n/a | 2.212 | n/a | n/a | 0.518 | n/a | n/a | 0.802 | 0.829 | -0.027 |
| L21 | -0.296 | n/a | n/a | 2.219 | n/a | n/a | 0.494 | n/a | n/a | 0.789 | 0.813 | -0.024 |
| L22 | -0.274 | n/a | n/a | 2.140 | n/a | n/a | 0.515 | n/a | n/a | 0.778 | 0.798 | -0.020 |
| L23 | -0.253 | n/a | n/a | 2.091 | n/a | n/a | 0.554 | n/a | n/a | 0.748 | 0.789 | -0.041 |
| L24 | -0.231 | n/a | n/a | 2.045 | n/a | n/a | 0.552 | n/a | n/a | 0.711 | 0.786 | -0.074 |
| L25 | -0.220 | n/a | n/a | 2.057 | n/a | n/a | 0.534 | n/a | n/a | 0.666 | 0.761 | -0.096 |
| L26 | -0.221 | n/a | n/a | 2.059 | n/a | n/a | 0.530 | n/a | n/a | 0.667 | 0.751 | -0.085 |
| L27 | -0.136 | n/a | n/a | 1.839 | n/a | n/a | 0.650 | n/a | n/a | 0.665 | 0.773 | -0.108 |

## Group-level layer curves — full span (clean only)

| group | class | transitional | strongest content layer | strongest procedure layer | margin curve |
| --- | --- | --- | ---: | ---: | --- |
| async_fastapi_service_design | content_like | no | 0 | 18 | L00:0.573 L01:0.563 L02:0.471 L03:0.471 L04:0.519 L05:0.509 L06:0.484 L07:0.489 L08:0.468 L09:0.459 L10:0.458 L11:0.458 L12:0.455 L13:0.424 L14:0.445 L15:0.444 L16:0.432 L17:0.425 L18:0.376 L19:0.440 L20:0.443 L21:0.480 L22:0.475 L23:0.483 L24:0.472 L25:0.453 L26:0.433 L27:0.480 |
| binary_search_update_loop | procedure_like | no | 18 | 1 | L00:-0.521 L01:-0.556 L02:-0.538 L03:-0.517 L04:-0.535 L05:-0.530 L06:-0.523 L07:-0.510 L08:-0.478 L09:-0.489 L10:-0.450 L11:-0.451 L12:-0.444 L13:-0.413 L14:-0.419 L15:-0.413 L16:-0.408 L17:-0.377 L18:-0.326 L19:-0.392 L20:-0.372 L21:-0.381 L22:-0.396 L23:-0.424 L24:-0.428 L25:-0.444 L26:-0.405 L27:-0.351 |
| dependency_injection_request_flow | procedure_like | no | 18 | 4 | L00:-0.490 L01:-0.458 L02:-0.487 L03:-0.469 L04:-0.495 L05:-0.480 L06:-0.467 L07:-0.434 L08:-0.429 L09:-0.438 L10:-0.421 L11:-0.417 L12:-0.413 L13:-0.370 L14:-0.383 L15:-0.397 L16:-0.391 L17:-0.396 L18:-0.363 L19:-0.387 L20:-0.388 L21:-0.428 L22:-0.447 L23:-0.424 L24:-0.416 L25:-0.414 L26:-0.388 L27:-0.478 |
| json_only_response_format | content_like | no | 27 | 18 | L00:0.451 L01:0.401 L02:0.428 L03:0.372 L04:0.341 L05:0.326 L06:0.327 L07:0.332 L08:0.329 L09:0.320 L10:0.321 L11:0.328 L12:0.321 L13:0.287 L14:0.292 L15:0.292 L16:0.286 L17:0.274 L18:0.239 L19:0.273 L20:0.273 L21:0.288 L22:0.279 L23:0.314 L24:0.320 L25:0.352 L26:0.329 L27:0.478 |
| proof_by_contradiction_method | procedure_like | no | 27 | 3 | L00:-0.413 L01:-0.395 L02:-0.421 L03:-0.426 L04:-0.397 L05:-0.361 L06:-0.339 L07:-0.309 L08:-0.322 L09:-0.317 L10:-0.325 L11:-0.315 L12:-0.306 L13:-0.296 L14:-0.284 L15:-0.281 L16:-0.285 L17:-0.297 L18:-0.276 L19:-0.274 L20:-0.298 L21:-0.335 L22:-0.306 L23:-0.295 L24:-0.311 L25:-0.317 L26:-0.313 L27:-0.262 |
| strictly_vegan_meal_plan | content_like | no | 2 | 27 | L00:0.520 L01:0.525 L02:0.589 L03:0.585 L04:0.579 L05:0.560 L06:0.556 L07:0.506 L08:0.486 L09:0.493 L10:0.442 L11:0.460 L12:0.442 L13:0.414 L14:0.412 L15:0.416 L16:0.429 L17:0.428 L18:0.394 L19:0.420 L20:0.418 L21:0.462 L22:0.487 L23:0.471 L24:0.486 L25:0.495 L26:0.454 L27:0.150 |

## Group-level layer curves — trimmed span (clean only)

| group | class | transitional | strongest content layer | strongest procedure layer | margin curve |
| --- | --- | --- | ---: | ---: | --- |
| async_fastapi_service_design | content_like | no | 23 | 18 | L00:0.570 L01:0.576 L02:0.567 L03:0.531 L04:0.543 L05:0.551 L06:0.554 L07:0.529 L08:0.512 L09:0.514 L10:0.539 L11:0.509 L12:0.518 L13:0.530 L14:0.539 L15:0.528 L16:0.511 L17:0.504 L18:0.478 L19:0.552 L20:0.549 L21:0.573 L22:0.604 L23:0.611 L24:0.594 L25:0.574 L26:0.565 L27:0.602 |
| binary_search_update_loop | procedure_like | no | 26 | 5 | L00:-0.662 L01:-0.646 L02:-0.646 L03:-0.631 L04:-0.636 L05:-0.687 L06:-0.661 L07:-0.648 L08:-0.655 L09:-0.665 L10:-0.683 L11:-0.636 L12:-0.623 L13:-0.594 L14:-0.622 L15:-0.605 L16:-0.593 L17:-0.564 L18:-0.535 L19:-0.580 L20:-0.544 L21:-0.521 L22:-0.499 L23:-0.483 L24:-0.424 L25:-0.401 L26:-0.355 L27:-0.362 |
| dependency_injection_request_flow | procedure_like | no | 26 | 9 | L00:-0.528 L01:-0.546 L02:-0.573 L03:-0.553 L04:-0.551 L05:-0.512 L06:-0.565 L07:-0.536 L08:-0.578 L09:-0.607 L10:-0.602 L11:-0.555 L12:-0.538 L13:-0.502 L14:-0.547 L15:-0.519 L16:-0.510 L17:-0.492 L18:-0.464 L19:-0.514 L20:-0.494 L21:-0.513 L22:-0.507 L23:-0.474 L24:-0.423 L25:-0.384 L26:-0.374 L27:-0.389 |
| json_only_response_format | content_like | no | 10 | 27 | L00:0.527 L01:0.476 L02:0.524 L03:0.492 L04:0.507 L05:0.517 L06:0.566 L07:0.549 L08:0.564 L09:0.571 L10:0.590 L11:0.564 L12:0.533 L13:0.501 L14:0.521 L15:0.503 L16:0.483 L17:0.483 L18:0.427 L19:0.434 L20:0.399 L21:0.368 L22:0.349 L23:0.339 L24:0.351 L25:0.353 L26:0.318 L27:0.168 |
| proof_by_contradiction_method | procedure_like | no | 19 | 24 | L00:-0.406 L01:-0.363 L02:-0.408 L03:-0.437 L04:-0.444 L05:-0.458 L06:-0.432 L07:-0.386 L08:-0.339 L09:-0.335 L10:-0.325 L11:-0.335 L12:-0.302 L13:-0.345 L14:-0.275 L15:-0.303 L16:-0.297 L17:-0.292 L18:-0.268 L19:-0.254 L20:-0.277 L21:-0.293 L22:-0.333 L23:-0.385 L24:-0.498 L25:-0.493 L26:-0.468 L27:-0.303 |
| strictly_vegan_meal_plan | content_like | no | 0 | 27 | L00:0.601 L01:0.584 L02:0.590 L03:0.600 L04:0.573 L05:0.598 L06:0.544 L07:0.521 L08:0.479 L09:0.531 L10:0.512 L11:0.451 L12:0.426 L13:0.429 L14:0.443 L15:0.442 L16:0.445 L17:0.417 L18:0.404 L19:0.424 L20:0.422 L21:0.410 L22:0.413 L23:0.399 L24:0.440 L25:0.441 L26:0.411 L27:0.094 |

## Full-span vs trimmed-span comparison

| subset | verdict | case_count | max separation layer | first positive layer | stable birth layer |
| --- | --- | ---: | ---: | ---: | ---: |
| full_span / all_valid | partial_separation | 12 | 26 | 3 | n/a |
| full_span / clean_only | partial_separation | 12 | 26 | 3 | n/a |
| trimmed_span / all_valid | no_separation | 12 | 2 | 2 | n/a |
| trimmed_span / clean_only | no_separation | 12 | 2 | 2 | n/a |

## Layer of maximal separation

- Full span / clean only: `{'layer': 26, 'separation_score': 0.75, 'positive_signals': 3, 'available_signals': 4}`
- Trimmed span / clean only: `{'layer': 2, 'separation_score': 1.0, 'positive_signals': 1, 'available_signals': 1}`

## Whether evidence supports polarity-from-geometry

Current judgment: `not_supported`.

The strongest evidence should come from clean-only trimmed spans, because that setting removes the first-token boundary artifact while keeping the original tokenizer intact. If full-span and trimmed-span disagree, trimmed-span is treated as the stricter control.

## Limitations

- The prompt set is still small and local, so group-level conclusions can be noisy.
- Trimming removes one token from every anchor; in this prompt set only the 5-token FastAPI group keeps enough tokens for coherence, tortuosity, and rank-1 EV after trimming.
- Cross-prompt stability is defined within the paired prompts already present in the probe, not over a large paraphrase set.
- Mean-direction affinity is a diagnostic proxy and not a causal proof of polarity.
- Some groups may remain transitional because the phrase itself mixes content and procedure semantics.

## Recommended next step

Run a tightly paired paraphrase probe per anchor group, holding token count fixed while varying only the surrounding sentence frame.
