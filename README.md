# Pakistan Flood Intelligence

An AI-assisted decision-support prototype for **planning and preparing resources before, during and after a flood in Pakistan**. It estimates flood risk, ranks damaged assets by recovery priority, and builds a repair plan that fits a budget and the available workers and equipment.

> **Prototype only.** The recovery records are synthetic asset-level examples calibrated to NDMA 2025 infrastructure damage information, and costs and resource needs are estimates. Nothing here is an official flood warning, engineering assessment or government recovery decision.

## What the app does

The app has four tabs.

| Tab | What you can do |
|---|---|
| **Predict Flood Risk** | Pick a province and area, enter expected rainfall and humidity, and get a flood-risk estimate from a model trained on historical flood records and tested on records it had not seen. |
| **Plan Recovery** | See assets ranked by AI priority, set a budget (and optionally worker and equipment limits), and get a recommended repair plan with a schedule, a coverage gap, an explanation of why the top assets were chosen, a cost-uncertainty check, a budget what-if table and an equity check. You can also **add assets or upload a CSV** of updated data. |
| **Future Resource Planning** | Explore Low, Moderate and Severe flood scenarios for a planning year, with a preparedness plan and indicative relief supplies (water, food, tents). |
| **Dataset & Model** | Browse the data and see how well each model performs on records it had not seen. |

An optional **AI situation briefing** appears under each plan. Without an API key it shows a standard written summary built from the computed figures. With a key it is written by an AI model, but all numbers still come from the planning code.

## Files

```
app.py                                        the Streamlit app
requirements.txt                              packages to install
Pakistan_flood_dataset.csv                    historical flood data (add this file here)
pakistan_asset_recovery_2025_prototype.csv    prototype recovery data (add this file here)
sample_data/                                  CSVs for trying the upload feature
Netsol_Project_Demo.ipynb                     Colab notebook that runs the app
.github/workflows/ci.yml                      automatic check on every push
```

Keep the two dataset files in the top folder. The app reads them from there; if they are not present it downloads them from this repository on GitHub.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run it in Google Colab

Open `Netsol_Project_Demo.ipynb` in Colab and run the cells from top to bottom. The last cell prints a public link to the app.

## Try the upload feature

Open **Plan Recovery**, then **Add or upload assets**.

- `sample_data/dummy_assets_clean.csv` has 30 test assets that all load correctly.
- `sample_data/dummy_assets_messy.csv` has deliberate problems (bad ratings, a negative cost, a duplicate ID, unusual values) so you can see how the app reports them.
- `sample_data/asset_upload_template.csv` shows the expected layout. Required columns: `province`, `area_name`, `asset_type`, `damage_severity_1_5`, `criticality_1_5`, `accessibility_1_5`, `population_served_or_affected_est`, `flood_impact_pct_est`, `estimated_repair_cost_pkr`. Workers, equipment and repair days are optional and estimated if left blank.

Added assets stay in your browser session only and are lost when the page is refreshed.

## Optional: AI-written briefings

Set an `ANTHROPIC_API_KEY` as an environment variable, in Colab Secrets, or in Streamlit's secrets (`.streamlit/secrets.toml`). **Never commit a key to GitHub.** The plan summary (totals and the five highest-priority assets) is sent to the AI service to write the briefing. Without a key everything else works unchanged.

## Automatic checks

`.github/workflows/ci.yml` runs on every push and pull request. It installs the packages, checks `app.py` for errors, then opens the app and clicks through the main buttons.

## Limitations

- The recovery priority labels are derived from the same inputs the model sees, so the model mainly learns that scoring rule. With real field-assessed priorities, the same pipeline would learn from actual decisions.
- Future scenarios are what-if calculations, not forecasts.
- Relief supply figures use editable planning assumptions (only the 15 litres of water per person per day is a standard humanitarian figure).
