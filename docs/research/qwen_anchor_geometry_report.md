# Qwen Anchor Geometry Report

Date: 2026-04-01 08:49 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max length: `128`
Representative layers: `[7, 14, 21, 27]`

## Research question

Can anchor polarity be inferred from the geometry of hidden-state assembly across the anchor span?

## Design

- Prompt count: `12`
- Valid prompts: `12`
- Skipped prompts: `0`
- Geometry is computed from raw Qwen hidden states obtained through the repo overlay loader, without generation steering.
- Deltas are token-to-token differences inside the matched anchor span.
- `rank1_explained_variance` is implemented as the rank-1 energy fraction of the uncentered delta matrix.

## Overall summary

- Verdict: `partial_separation`
- Positive signals: `9` / `16`
- Recommended next step: `refine_metric`

## Layer-by-layer class comparison

| Layer | Content coherence | Procedure coherence | Content tortuosity | Procedure tortuosity | Content rank-1 EV | Procedure rank-1 EV | Content stability | Procedure stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | -0.4470 | -0.4689 | 3.1316 | 2.9252 | 0.5609 | 0.5883 | 0.3054 | 0.2660 |
| 14 | -0.4207 | -0.4635 | 2.9153 | 2.7282 | 0.5355 | 0.5930 | 0.3782 | 0.3362 |
| 21 | -0.3908 | -0.4625 | 2.7388 | 2.5769 | 0.5512 | 0.6003 | 0.3740 | 0.3035 |
| 27 | -0.3334 | -0.4290 | 2.6926 | 2.7380 | 0.5522 | 0.6393 | 0.3463 | 0.2495 |

## Per-prompt table

| Class | Group | Case | Layer | Tokens | Deltas | Coherence | Tortuosity | Rank-1 EV | Mean dir norm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| content_like | strictly_vegan_meal_plan | content_vegan_brief | 7 | 4 | 3 | -0.4154 | 2.5146 | 0.5998 | 19.8938 |
| content_like | strictly_vegan_meal_plan | content_vegan_brief | 14 | 4 | 3 | -0.3989 | 2.5428 | 0.5894 | 21.5768 |
| content_like | strictly_vegan_meal_plan | content_vegan_brief | 21 | 4 | 3 | -0.3569 | 2.3506 | 0.5696 | 35.8799 |
| content_like | strictly_vegan_meal_plan | content_vegan_brief | 27 | 4 | 3 | -0.2916 | 2.3116 | 0.5465 | 70.3580 |
| content_like | strictly_vegan_meal_plan | content_vegan_reason | 7 | 4 | 3 | -0.4291 | 2.5435 | 0.5970 | 20.3396 |
| content_like | strictly_vegan_meal_plan | content_vegan_reason | 14 | 4 | 3 | -0.4076 | 2.4396 | 0.5868 | 22.5314 |
| content_like | strictly_vegan_meal_plan | content_vegan_reason | 21 | 4 | 3 | -0.3740 | 2.2428 | 0.5643 | 38.1308 |
| content_like | strictly_vegan_meal_plan | content_vegan_reason | 27 | 4 | 3 | -0.3323 | 2.2876 | 0.5568 | 72.8001 |
| content_like | async_fastapi_service_design | content_fastapi_architecture | 7 | 5 | 4 | -0.4280 | 3.8038 | 0.4784 | 14.0657 |
| content_like | async_fastapi_service_design | content_fastapi_architecture | 14 | 5 | 4 | -0.4117 | 3.5721 | 0.4237 | 16.2950 |
| content_like | async_fastapi_service_design | content_fastapi_architecture | 21 | 5 | 4 | -0.4073 | 3.4040 | 0.4954 | 24.9570 |
| content_like | async_fastapi_service_design | content_fastapi_architecture | 27 | 5 | 4 | -0.4112 | 3.6315 | 0.5818 | 45.7660 |
| content_like | async_fastapi_service_design | content_fastapi_summary | 7 | 5 | 4 | -0.4364 | 3.9722 | 0.5061 | 12.9151 |
| content_like | async_fastapi_service_design | content_fastapi_summary | 14 | 5 | 4 | -0.3987 | 3.4091 | 0.4606 | 16.1713 |
| content_like | async_fastapi_service_design | content_fastapi_summary | 21 | 5 | 4 | -0.3857 | 3.3371 | 0.5427 | 25.1809 |
| content_like | async_fastapi_service_design | content_fastapi_summary | 27 | 5 | 4 | -0.3569 | 3.3957 | 0.5672 | 45.3870 |
| content_like | json_only_response_format | content_json_contract | 7 | 4 | 3 | -0.4856 | 3.0043 | 0.5928 | 15.2467 |
| content_like | json_only_response_format | content_json_contract | 14 | 4 | 3 | -0.4594 | 2.7997 | 0.5798 | 18.7082 |
| content_like | json_only_response_format | content_json_contract | 21 | 4 | 3 | -0.4081 | 2.4697 | 0.5544 | 29.6534 |
| content_like | json_only_response_format | content_json_contract | 27 | 4 | 3 | -0.2467 | 2.0835 | 0.4656 | 58.1741 |
| content_like | json_only_response_format | content_json_parser | 7 | 4 | 3 | -0.4877 | 2.9510 | 0.5914 | 15.9150 |
| content_like | json_only_response_format | content_json_parser | 14 | 4 | 3 | -0.4477 | 2.7285 | 0.5726 | 19.6391 |
| content_like | json_only_response_format | content_json_parser | 21 | 4 | 3 | -0.4129 | 2.6286 | 0.5807 | 29.3675 |
| content_like | json_only_response_format | content_json_parser | 27 | 4 | 3 | -0.3616 | 2.4459 | 0.5951 | 57.3568 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_proof | 7 | 4 | 3 | -0.4707 | 3.3392 | 0.5508 | 14.2901 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_proof | 14 | 4 | 3 | -0.4919 | 3.0021 | 0.5860 | 18.6201 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_proof | 21 | 4 | 3 | -0.4294 | 2.5289 | 0.6044 | 31.2772 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_proof | 27 | 4 | 3 | -0.4416 | 3.0905 | 0.7775 | 57.2178 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_explain | 7 | 4 | 3 | -0.4856 | 3.2715 | 0.5769 | 14.0781 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_explain | 14 | 4 | 3 | -0.5057 | 3.1111 | 0.6016 | 17.7874 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_explain | 21 | 4 | 3 | -0.5029 | 3.1061 | 0.6554 | 25.8974 |
| procedure_like | proof_by_contradiction_method | procedure_contradiction_explain | 27 | 4 | 3 | -0.5113 | 3.9134 | 0.8260 | 45.4844 |
| procedure_like | binary_search_update_loop | procedure_binary_search_note | 7 | 4 | 3 | -0.4688 | 2.7189 | 0.6038 | 17.7289 |
| procedure_like | binary_search_update_loop | procedure_binary_search_note | 14 | 4 | 3 | -0.4169 | 2.4794 | 0.5789 | 22.2082 |
| procedure_like | binary_search_update_loop | procedure_binary_search_note | 21 | 4 | 3 | -0.4322 | 2.4017 | 0.5707 | 33.9978 |
| procedure_like | binary_search_update_loop | procedure_binary_search_note | 27 | 4 | 3 | -0.3513 | 2.0875 | 0.5288 | 62.9813 |
| procedure_like | binary_search_update_loop | procedure_binary_search_indices | 7 | 4 | 3 | -0.4923 | 2.8602 | 0.6236 | 17.5124 |
| procedure_like | binary_search_update_loop | procedure_binary_search_indices | 14 | 4 | 3 | -0.4558 | 2.6065 | 0.6001 | 21.7116 |
| procedure_like | binary_search_update_loop | procedure_binary_search_indices | 21 | 4 | 3 | -0.4318 | 2.3756 | 0.5636 | 35.6090 |
| procedure_like | binary_search_update_loop | procedure_binary_search_indices | 27 | 4 | 3 | -0.3700 | 2.1373 | 0.5482 | 70.1049 |
| procedure_like | dependency_injection_request_flow | procedure_di_request_path | 7 | 4 | 3 | -0.4411 | 2.6818 | 0.5688 | 17.0541 |
| procedure_like | dependency_injection_request_flow | procedure_di_request_path | 14 | 4 | 3 | -0.4381 | 2.5863 | 0.5753 | 20.4177 |
| procedure_like | dependency_injection_request_flow | procedure_di_request_path | 21 | 4 | 3 | -0.5085 | 2.6401 | 0.6010 | 30.9225 |
| procedure_like | dependency_injection_request_flow | procedure_di_request_path | 27 | 4 | 3 | -0.4776 | 2.7115 | 0.5643 | 53.0202 |
| procedure_like | dependency_injection_request_flow | procedure_di_summary | 7 | 4 | 3 | -0.4551 | 2.6795 | 0.6059 | 17.6072 |
| procedure_like | dependency_injection_request_flow | procedure_di_summary | 14 | 4 | 3 | -0.4726 | 2.5836 | 0.6162 | 21.0323 |
| procedure_like | dependency_injection_request_flow | procedure_di_summary | 21 | 4 | 3 | -0.4704 | 2.4093 | 0.6066 | 33.2698 |
| procedure_like | dependency_injection_request_flow | procedure_di_summary | 27 | 4 | 3 | -0.4222 | 2.4878 | 0.5911 | 54.7819 |

## Cross-prompt stability by anchor group

| Group | Layer | Pair count | Mean pairwise cosine |
|---|---:|---:|---:|
| async_fastapi_service_design | 7 | 1 | 0.8754 |
| async_fastapi_service_design | 14 | 1 | 0.8794 |
| async_fastapi_service_design | 21 | 1 | 0.8765 |
| async_fastapi_service_design | 27 | 1 | 0.7877 |
| binary_search_update_loop | 7 | 1 | 0.8922 |
| binary_search_update_loop | 14 | 1 | 0.8800 |
| binary_search_update_loop | 21 | 1 | 0.8471 |
| binary_search_update_loop | 27 | 1 | 0.6618 |
| dependency_injection_request_flow | 7 | 1 | 0.8852 |
| dependency_injection_request_flow | 14 | 1 | 0.8787 |
| dependency_injection_request_flow | 21 | 1 | 0.9147 |
| dependency_injection_request_flow | 27 | 1 | 0.8583 |
| json_only_response_format | 7 | 1 | 0.9503 |
| json_only_response_format | 14 | 1 | 0.8844 |
| json_only_response_format | 21 | 1 | 0.8100 |
| json_only_response_format | 27 | 1 | 0.7306 |
| proof_by_contradiction_method | 7 | 1 | 0.8630 |
| proof_by_contradiction_method | 14 | 1 | 0.7965 |
| proof_by_contradiction_method | 21 | 1 | 0.7077 |
| proof_by_contradiction_method | 27 | 1 | 0.5152 |
| strictly_vegan_meal_plan | 7 | 1 | 0.9196 |
| strictly_vegan_meal_plan | 14 | 1 | 0.8472 |
| strictly_vegan_meal_plan | 21 | 1 | 0.7761 |
| strictly_vegan_meal_plan | 27 | 1 | 0.5677 |

## Interpretation

- Layer 7: supporting signals = coherence, stability.
- Layer 14: supporting signals = coherence, stability.
- Layer 21: supporting signals = coherence, stability.
- Layer 27: supporting signals = coherence, tortuosity, stability.

## Limitations

- The prompt set is intentionally small and local, so class-level separation can still be confounded by phrase identity.
- Content-like anchors remain semantically heterogeneous even after phrase grouping; phrase-level stability is therefore reported separately.
- Short spans would disable curvature-style metrics, although this prompt set was chosen to keep spans at four tokens in most cases.
- This experiment measures static hidden-state assembly during prompt processing only; it does not test runtime steering quality directly.
