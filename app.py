import streamlit as st
import pandas as pd
import numpy as np
import pulp
from pycaret.regression import load_model, predict_model
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 🔥 Load Master Data and Trained ML Model
# ============================================
MASTER_FILE_PATH = 'main_file_updated.csv'
MODEL_PATH = 'ml_avg_predictor_model'

# --- Add Name Corrections Dictionary ---
NAME_CORRECTIONS = {
    
}
# ---
@st.cache_data
def load_master_data():
    try:
        df = pd.read_csv(MASTER_FILE_PATH)
        # Apply column name cleaning
        df.columns = df.columns.str.replace('  ', ' ', regex=False).str.strip()
        # Apply Name Corrections after stripping
        df['Player Name Clean'] = df['Player Name'].str.strip()
        df['Player Name Clean'] = df['Player Name Clean'].replace(NAME_CORRECTIONS)
        # Replace infinite values and handle missing data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if 'Dream11_Avg' in df.columns and 'Dream_11 round 1' in df.columns and 'Dream_11_2024_avg' in df.columns:
            df['Dream11_Avg'] = pd.to_numeric(df['Dream11_Avg'], errors='coerce').fillna(0)
            df['Dream_11 round 1'] = pd.to_numeric(df['Dream_11 round 1'], errors='coerce').fillna(0)
            df['Dream_11_2024_avg'] = pd.to_numeric(df['Dream_11_2024_avg'], errors='coerce').fillna(0)
            # Calculate Weighted_Score with new weightage
            df['Weighted_Score'] = (
                (0.60 * df['Dream11_Avg']) +
                (0.15 * df['Dream_11_2024_avg']) +
                (0.25 * df['Dream_11 round 1'])
            )
        else:
            df['Weighted_Score'] = 0
        return df
    except FileNotFoundError:
        st.error(f"❌ Master file not found: {MASTER_FILE_PATH}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading master file: {e}")
        return None

@st.cache_resource
def load_trained_ml_model():
    # ... (Function remains the same) ...
    model_file = MODEL_PATH + '.pkl'
    if not os.path.exists(model_file):
         st.error(f"❌ Trained model not found: {model_file}. Run TRAIN.py."); return None
    try: return load_model(MODEL_PATH)
    except Exception as e: st.error(f"❌ Error loading ML model: {e}"); return None

# ============================================
# 📝 Prepare Input Data (Apply Name Corrections, Add Debugging)
# ============================================
def prepare_data_for_prediction(match_players_df, master_df):
    """
    Prepares player list for ML prediction using specific name corrections and merge.
    Includes debugging for missing stats.
    """
    if master_df is None:
        return None, None

    career_features_expected = [
        'Innings', 'Runs', 'Balls Faced', 'Fours', 'Sixes', 'Batting Average',
        'Strike Rate', 'Boundary_Percentage', 'Innings_X', 'Balls Bowled',
        'Runs_Conceded', 'Wickets', 'Bowling Average', 'Economy Rate',
        'Bowling Strike_rate'
    ]
    match_features_expected = ['Credits', 'Player Type', 'Team', 'Weighted_Score']
    all_features_for_model = match_features_expected + career_features_expected
    available_master_career_features = [f for f in career_features_expected if f in master_df.columns]
    if len(available_master_career_features) != len(career_features_expected):
        return None, None

    required_match_cols = ['Player Name', 'Credits', 'Player Type', 'Team']
    optional_match_cols = ['IsPlaying', 'lineupOrder']
    present_optional_cols = [col for col in optional_match_cols if col in match_players_df.columns]
    if not all(col in match_players_df.columns for col in required_match_cols):
        return None, None

    players_for_match = match_players_df.copy()
    if 'IsPlaying' in present_optional_cols:
        playing_mask = players_for_match['IsPlaying'].str.upper().fillna('') == 'PLAYING'
        if playing_mask.any():
            players_for_match = players_for_match[playing_mask].copy()
        else:
            st.warning("No 'PLAYING' players found. Using all.")

    # Clean Name & Apply Corrections from Uploaded File
    players_for_match['Player Name Clean'] = players_for_match['Player Name'].str.strip()
    players_for_match['Player Name Clean'] = players_for_match['Player Name Clean'].replace(NAME_CORRECTIONS)

    if 'lineupOrder' in present_optional_cols:
        players_for_match['lineupOrder'] = pd.to_numeric(players_for_match['lineupOrder'], errors='coerce').fillna(99)
    else:
        players_for_match['lineupOrder'] = 99

    cols_to_select_from_match = ['Player Name', 'Player Name Clean', 'Credits', 'Player Type', 'Team', 'lineupOrder']
    prediction_input_base = players_for_match[
        [col for col in cols_to_select_from_match if col in players_for_match.columns]
    ].drop_duplicates(subset=['Player Name Clean']).copy()
    prediction_input_base['Credits'] = pd.to_numeric(prediction_input_base['Credits'], errors='coerce').fillna(0)

    # Select master features using the ALREADY CORRECTED 'Player Name Clean'
    master_features_to_merge = master_df[['Player Name Clean'] + available_master_career_features + ['Weighted_Score']].drop_duplicates(subset=['Player Name Clean'])

    prediction_input_merged = pd.merge(
        prediction_input_base,
        master_features_to_merge,
        on='Player Name Clean',  # Join on the corrected clean name
        how='left'
    )

    # Handle Missing Career Features & Dtypes
    missing_career_mask = prediction_input_merged[available_master_career_features].isnull().any(axis=1)
    if missing_career_mask.any():
        missing_players_df = prediction_input_merged.loc[missing_career_mask, ['Player Name', 'Player Name Clean']]
        missing_players_list = missing_players_df['Player Name'].tolist()
        st.warning(
            f"Players possibly missing career stats (imputing with median): {missing_players_list[:10]}"
            f"{'...' if len(missing_players_list) > 10 else ''}. "
            "Check master file for completeness for these players if name correction didn't fix it."
        )
        for col in available_master_career_features:
            if prediction_input_merged[col].isnull().any() and pd.api.types.is_numeric_dtype(master_df[col]):
                median_val = master_df[col].median()
                prediction_input_merged[col].fillna(median_val if pd.notna(median_val) else 0, inplace=True)

    # Dtype enforcement remains the same
    categorical_features_model = ['Player Type', 'Team']
    numeric_features_model = ['Credits'] + available_master_career_features + ['Weighted_Score']
    for col in numeric_features_model:
        if col in prediction_input_merged.columns:
            prediction_input_merged[col] = pd.to_numeric(prediction_input_merged[col], errors='coerce').fillna(0)
    for col in categorical_features_model:
        if col in prediction_input_merged.columns:
            prediction_input_merged[col] = prediction_input_merged[col].astype(str).fillna('Unknown')

    # Prepare Final DataFrames
    features_for_model_final = prediction_input_merged[
        [col for col in all_features_for_model if col in prediction_input_merged.columns]
    ].copy()
    info_for_optimization_final = prediction_input_merged[
        ['Player Name', 'Credits', 'Player Type', 'Team', 'lineupOrder', 'Weighted_Score']
    ].copy()

    return features_for_model_final, info_for_optimization_final

# ============================================
# 🏏 Select Best Team (Orchestrator using ML + PuLP)
# ============================================
def select_best_team(features_for_prediction, player_info_for_optimization, ml_model):
    # ... (Function remains the same) ...
    if features_for_prediction is None or features_for_prediction.empty: return pd.DataFrame()
    if player_info_for_optimization is None or player_info_for_optimization.empty: return pd.DataFrame()
    if ml_model is None: return pd.DataFrame()
    try:
        predictions_with_meta = predict_model(ml_model, data=features_for_prediction)
        if 'prediction_label' not in predictions_with_meta.columns: return pd.DataFrame()
        predicted_scores_only = predictions_with_meta[['prediction_label']].rename(columns={'prediction_label': 'Predicted_Score'})
        predicted_scores_only['Predicted_Score'] = predicted_scores_only['Predicted_Score'].clip(lower=0)
        # Use index for joining prediction scores to optimization info
        optimization_input = pd.concat([player_info_for_optimization.reset_index(drop=True), predicted_scores_only.reset_index(drop=True)], axis=1)
        optimization_input = optimization_input.dropna(subset=['Credits', 'Predicted_Score', 'Player Name', 'Team', 'Player Type', 'lineupOrder'])
        optimization_input['Credits'] = pd.to_numeric(optimization_input['Credits'], errors='coerce').fillna(0)
        optimization_input = optimization_input[optimization_input['Credits'] > 0]
        if optimization_input.empty: return pd.DataFrame()
        selected_team_df = select_best_team_optimized_pulp(optimization_input, max_credits=100)
        return selected_team_df
    except Exception as e:
        st.error(f"❌ Error during ML prediction or optimization prep: {e}")
        import traceback; st.error(traceback.format_exc())
        return pd.DataFrame()

# ============================================
#  Optimizer Function (PuLP with Lineup Preference)
# ============================================
def select_best_team_optimized_pulp(players_df, max_credits=100, lineup_bonus_factor=0.2):
    # ... (PuLP function remains the same) ...
    if players_df is None or players_df.empty: return pd.DataFrame()
    players_list = players_df['Player Name'].tolist()
    credits = players_df.set_index('Player Name')['Credits'].to_dict()
    predicted_scores = players_df.set_index('Player Name')['Predicted_Score'].to_dict()
    roles_raw = players_df.set_index('Player Name')['Player Type'].to_dict()
    teams = players_df.set_index('Player Name')['Team'].to_dict()
    lineup_orders = players_df.set_index('Player Name')['lineupOrder'].to_dict()
    adjusted_scores = {}
    for p in players_list:
        base_score = predicted_scores.get(p, 0)
        role = str(roles_raw.get(p, '')).upper().strip()
        order = lineup_orders.get(p, 99)
        is_batsman = 'BAT' in role and 'WICKETKEEPER' not in role
        bonus = 0
        if is_batsman and 1 <= order <= 5:
             bonus = base_score * lineup_bonus_factor * ( (5 - order) / 5.0 )
        adjusted_scores[p] = base_score + bonus
    roles = {name: str(role).upper().strip() for name, role in roles_raw.items()}
    player_vars = pulp.LpVariable.dicts("Player", players_list, cat='Binary')
    prob = pulp.LpProblem("Dream11_Team_Selection_ML_Lineup", pulp.LpMaximize)
    prob += pulp.lpSum(adjusted_scores[p] * player_vars[p] for p in players_list if p in adjusted_scores), "Total_Adjusted_Score"
    prob += pulp.lpSum(credits[p] * player_vars[p] for p in players_list if p in credits) <= max_credits, "Max_Credits"
    prob += pulp.lpSum(player_vars[p] for p in players_list) == 11, "Total_Players"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('WK' in roles[p] or 'WICKETKEEPER' in roles[p])) >= 1, "Min_WK"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('WK' in roles[p] or 'WICKETKEEPER' in roles[p])) <= 4, "Max_WK"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('BAT' in roles[p] and 'WICKETKEEPER' not in roles[p])) >= 1, "Min_BAT_Pure"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('BAT' in roles[p] and 'WICKETKEEPER' not in roles[p])) <= 6, "Max_BAT_Pure"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('ALL' in roles[p] or 'ROUNDER' in roles[p])) >= 1, "Min_AR"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('ALL' in roles[p] or 'ROUNDER' in roles[p])) <= 6, "Max_AR"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('BOWL' in roles[p])) >= 1, "Min_BOWL"
    prob += pulp.lpSum(player_vars[p] for p in players_list if p in roles and ('BOWL' in roles[p])) <= 6, "Max_BOWL"
    unique_teams = players_df['Team'].unique()
    for team_name in unique_teams:
        if pd.notna(team_name):
            prob += pulp.lpSum(player_vars[p] for p in players_list if p in teams and teams[p] == team_name) <= 10, f"Max_{str(team_name).replace(' ','_').replace('/','')}"
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    solve_status = pulp.LpStatus[prob.status]
    if solve_status != 'Optimal': return pd.DataFrame()
    selected_player_names = [p for p in players_list if player_vars[p].varValue > 0.5]
    selected_team_df = players_df[players_df['Player Name'].isin(selected_player_names)].copy()
    selected_team_df['Score_Used'] = selected_team_df['Predicted_Score']
    if len(selected_team_df) != 11: st.warning(f"Optimizer selected {len(selected_team_df)} players.")
    return selected_team_df


# ============================================
# 👑 Assign Captain and Vice-Captain
# ============================================
def assign_captain_vice_captain(players_df):
    # ... (Function remains the same) ...
    if players_df is None or players_df.empty or 'Score_Used' not in players_df.columns: return pd.DataFrame()
    sorted_players = players_df.sort_values('Score_Used', ascending=False).copy()
    sorted_players['C/VC'] = ''
    if len(sorted_players) >= 1:
        captain_index = sorted_players.index[0]
        sorted_players.loc[captain_index, 'C/VC'] = 'C'
        if len(sorted_players) >= 2:
            if 'Player Type' in sorted_players.columns:
                captain_role = str(sorted_players.loc[captain_index, 'Player Type']).upper().strip()
                potential_vc = sorted_players[(sorted_players.index != captain_index) & (sorted_players['Player Type'].astype(str).str.upper().str.strip() != captain_role)]
                if not potential_vc.empty: vc_index = potential_vc.index[0]
                else: vc_index = sorted_players.index[1]
                sorted_players.loc[vc_index, 'C/VC'] = 'VC'
            else:
                 vc_index = sorted_players.index[1]
                 sorted_players.loc[vc_index, 'C/VC'] = 'VC'
    return sorted_players

# ============================================
# 🚀 Streamlit App (Cleaned UI)
# ============================================
# ... (UI Section remains the same as the previous version) ...
st.set_page_config(layout="wide")
st.title("🏆 DREAM 11 TEAM PREDICTOR")
st.write("Upload match CSV. Predicts score using ML, applies lineup preference, then optimizes team selection.")

master_data = load_master_data()
ml_model = load_trained_ml_model()

uploaded_file = st.file_uploader("📄 Upload Match CSV", type=['csv'])

if uploaded_file is not None and master_data is not None and ml_model is not None:
    try:
        match_data_upload = pd.read_csv(uploaded_file)

        features_for_ml, info_for_opt = prepare_data_for_prediction(match_data_upload, master_data)

        if features_for_ml is not None and info_for_opt is not None:
            best_team = select_best_team(features_for_ml, info_for_opt, ml_model)

            if not best_team.empty:
                final_team = assign_captain_vice_captain(best_team)

                st.write("---")
                st.subheader("✅ Predicted Best XI")
                total_credits_used = final_team['Credits'].sum()
                st.metric("Credits Used", f"{total_credits_used:.1f} / 100", delta=f"{100-total_credits_used:.1f} Remaining", delta_color="off")

                output_columns = ['Player Name', 'Team', 'C/VC']
                display_team = final_team[output_columns].sort_values(by=['C/VC','Player Name'], ascending=[True, True])
                st.dataframe(display_team, use_container_width=True)

                csv = display_team.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Team CSV", csv,
                    f"predicted_team_{uploaded_file.name.split('.')[0]}.csv",
                    "text/csv", key='download-final-csv'
                )
            else:
                st.error("🚨 Could not generate a team after optimization.")
        else:
             st.error("🚨 Data preparation failed. Check input files and column names.")

    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        import traceback; st.error(f"Traceback: {traceback.format_exc()}")

elif master_data is None: st.error("🚨 Master data could not be loaded.")
elif ml_model is None: st.error("🚨 ML Model could not be loaded.")
else: st.warning("⚠️ Please upload a CSV file to begin.")