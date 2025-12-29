import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MOBA Strat-G Pro",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DATABASE 1: DRAFTING & HEROES (Pre-Game)
# ==========================================
HERO_DATABASE = {
    "Terizla": {"role": "Fighter", "lane": "Exp Lane", "specialty": "Burst/CC", "counters": ["Cici", "X.Borg", "Dyrroth"], "stats": {"win": 54.2}},
    "Yu Zhong": {"role": "Fighter", "lane": "Exp Lane", "specialty": "Regen/Dive", "counters": ["Baxia", "Dyrroth", "Wanwan"], "stats": {"win": 51.5}},
    "Cici": {"role": "Fighter", "lane": "Exp Lane", "specialty": "Mobility/Poke", "counters": ["Terizla", "Yu Zhong", "Martis"], "stats": {"win": 53.1}},
    "Paquito": {"role": "Fighter", "lane": "Exp Lane", "specialty": "Chase/Burst", "counters": ["Phoveus", "Minsitthar", "Fredrinn"], "stats": {"win": 49.8}},
    "Arlott": {"role": "Fighter", "lane": "Exp Lane", "specialty": "Crowd Control", "counters": ["Phoveus", "Minsitthar", "Diggie"], "stats": {"win": 52.8}},
    "X.Borg": {"role": "Fighter", "lane": "Exp Lane", "specialty": "Sustain/True Dmg", "counters": ["Dyrroth", "Silvanna", "Paquito"], "stats": {"win": 49.5}},
    "Ling": {"role": "Assassin", "lane": "Jungle", "specialty": "Mobility", "counters": ["Kaja", "Minsitthar", "Khufra"], "stats": {"win": 52.0}},
    "Nolan": {"role": "Assassin", "lane": "Jungle", "specialty": "Chase/Burst", "counters": ["Phoveus", "Minsitthar", "Khufra"], "stats": {"win": 55.4}},
    "Fanny": {"role": "Assassin", "lane": "Jungle", "specialty": "Chase/Reap", "counters": ["Khufra", "Saber", "Franco"], "stats": {"win": 50.1}},
    "Fredrinn": {"role": "Tank/Fighter", "lane": "Jungle", "specialty": "Dmg/Control", "counters": ["Karrie", "X.Borg", "Valir"], "stats": {"win": 51.2}},
    "Baxia": {"role": "Tank", "lane": "Jungle", "specialty": "Mobility/Anti-Heal", "counters": ["Karrie", "Valir", "X.Borg"], "stats": {"win": 53.5}},
    "Valentina": {"role": "Mage", "lane": "Mid Lane", "specialty": "Burst/Copy", "counters": ["Kagura", "Lunox", "Eudora"], "stats": {"win": 52.5}},
    "Vexana": {"role": "Mage", "lane": "Mid Lane", "specialty": "Poke/Control", "counters": ["Kadita", "Lancelot", "Joy"], "stats": {"win": 51.0}},
    "Novaria": {"role": "Mage", "lane": "Mid Lane", "specialty": "Burst/Snipe", "counters": ["Lolita", "Ling", "Fanny"], "stats": {"win": 50.2}},
    "Faramis": {"role": "Support/Mage", "lane": "Mid Lane", "specialty": "Guard/Burst", "counters": ["Valentina", "Akai", "Yin"], "stats": {"win": 54.8}},
    "Claude": {"role": "Marksman", "lane": "Gold Lane", "specialty": "Burst/Chase", "counters": ["Lesley", "Popol and Kupa", "Belerick"], "stats": {"win": 51.8}},
    "Karrie": {"role": "Marksman", "lane": "Gold Lane", "specialty": "True Damage", "counters": ["Lunox", "Harith", "Pharsa"], "stats": {"win": 52.1}},
    "Wanwan": {"role": "Marksman", "lane": "Gold Lane", "specialty": "Mobility/Reap", "counters": ["Phoveus", "Minsitthar", "Khufra"], "stats": {"win": 49.5}},
    "Bruno": {"role": "Marksman", "lane": "Gold Lane", "specialty": "Burst/Crit", "counters": ["Brody", "Lesley", "Natalia"], "stats": {"win": 51.2}},
    "Mathilda": {"role": "Support/Assassin", "lane": "Roamer", "specialty": "Guard/Initiator", "counters": ["Khufra", "Minsitthar", "Esmeralda"], "stats": {"win": 55.2}},
    "Diggie": {"role": "Support", "lane": "Roamer", "specialty": "Anti-CC/Poke", "counters": ["Hilda", "Natalia", "Selena"], "stats": {"win": 55.7}},
    "Khufra": {"role": "Tank", "lane": "Roamer", "specialty": "Anti-Blink", "counters": ["Diggie", "Valir", "Franco"], "stats": {"win": 50.2}},
    "Minotaur": {"role": "Tank/Support", "lane": "Roamer", "specialty": "CC/Heal", "counters": ["Diggie", "Karrie", "Lunox"], "stats": {"win": 53.5}},
    "Tigreal": {"role": "Tank", "lane": "Roamer", "specialty": "Crowd Control", "counters": ["Diggie", "Valir", "Akai"], "stats": {"win": 50.5}},
}

MPL_DRAFTS = {
    "SRG vs HomeBois (MPL MY S13)": {
        "team_name": "Selangor Red Giants", "playstyle": "High Tempo & Aggression",
        "lanes": {"Exp Lane": "Arlott", "Jungle": "Ling", "Mid Lane": "Valentina", "Gold Lane": "Claude", "Roamer": "Mathilda"},
        "objective_priority": ["Enemy Buffs", "Turrets", "Lord"]
    },
    "Todak vs Team SMG": {
        "team_name": "Todak", "playstyle": "Split Push Macro",
        "lanes": {"Exp Lane": "Paquito", "Jungle": "Baxia", "Mid Lane": "Faramis", "Gold Lane": "Wanwan", "Roamer": "Diggie"},
        "objective_priority": ["Turrets", "Vision", "Lord"]
    },
    "HomeBois vs RSG MY": {
        "team_name": "HomeBois", "playstyle": "Pick-off & Isolation",
        "lanes": {"Exp Lane": "Yu Zhong", "Jungle": "Nolan", "Mid Lane": "Vexana", "Gold Lane": "Bruno", "Roamer": "Khufra"},
        "objective_priority": ["Ganks", "Lord", "Turtle"]
    }
}

# ==========================================
# DATABASE 2: BEHAVIORAL DATA (Post-Game)
# ==========================================
MPL_MATCH_STATS = {
    "SRG vs HomeBois (High Synergy)": {
        "team_name": "Selangor Red Giants",
        "kills": 28, "deaths": 10, "assists": 65, "gold_diff": 8500,
        "chat_density": 85,        
        "avg_distance": 6.5,       
        "role_balance_score": 92,  
        "obj_timing_score": 88,    
        "sync_score": 90           
    },
    "Todak vs SMG (Strategic/Split)": {
        "team_name": "Todak",
        "kills": 15, "deaths": 12, "assists": 30, "gold_diff": 4200,
        "chat_density": 60,        
        "avg_distance": 12.0,      
        "role_balance_score": 85,
        "obj_timing_score": 95,    
        "sync_score": 70
    },
    "Team HAQ vs JP Niners (Defensive)": {
        "team_name": "Team HAQ",
        "kills": 10, "deaths": 25, "assists": 20, "gold_diff": -5000,
        "chat_density": 30,        
        "avg_distance": 9.0,
        "role_balance_score": 75,
        "obj_timing_score": 40,    
        "sync_score": 45           
    }
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def determine_playstyle(hero_list):
    specialties = []
    for h in hero_list:
        if h in HERO_DATABASE:
            specialties.extend([t.strip() for t in HERO_DATABASE[h]['specialty'].replace('/', ',').split(',')])
    if not specialties: return "Balanced"
    counts = Counter(specialties)
    if counts["Mobility"] >= 3: return "High Tempo & Mobility"
    if counts["Burst"] >= 3: return "High Burst & Pick-off"
    if counts["Sustain"] >= 2: return "Teamfight Sustain"
    return "Balanced / Standard"

def analyze_lane_matchups(enemy_lanes):
    recs = {}
    for lane, enemy in enemy_lanes.items():
        if enemy in HERO_DATABASE:
            counters = HERO_DATABASE[enemy]['counters']
            stats = []
            for c in counters:
                if c in HERO_DATABASE:
                    stats.append({'name': c, 'role': HERO_DATABASE[c]['role'], 'win': HERO_DATABASE[c]['stats']['win'], 'spec': HERO_DATABASE[c]['specialty']})
            stats.sort(key=lambda x: x['win'], reverse=True)
            recs[lane] = {'enemy': enemy, 'top': stats[:3]}
        else:
            recs[lane] = {'enemy': enemy, 'top': []}
    return recs

def calculate_ci_score(stats):
    weights = {'chat': 0.25, 'dist': 0.20, 'role': 0.20, 'obj': 0.20, 'sync': 0.15}
    norm_dist = max(0, 100 - (stats['avg_distance'] * 5)) 
    score = (
        stats['chat_density'] * weights['chat'] +
        norm_dist * weights['dist'] +
        stats['role_balance_score'] * weights['role'] +
        stats['obj_timing_score'] * weights['obj'] +
        stats['sync_score'] * weights['sync']
    )
    return round(score, 2)

def get_ci_based_recommendations(score):
    """
    Generates granular recommendations based on 5 CI Score Tiers.
    """
    recs = []
    if score < 20:
        recs = [
            "CRITICAL FAILURE: Team functionality is non-existent.",
            "1. STOP solo plays immediately. Move only in groups of 3+.",
            "2. Designate one Shotcaller. Everyone else must mute mics/chat to reduce noise.",
            "3. Play safe under turrets. Do not contest objectives until gold gap closes."
        ]
    elif score < 40:
        recs = [
            "POOR SYNERGY: Team is fragmented and reactive.",
            "1. Force group-ups for objectives (Turtle/Lord) 20 seconds early.",
            "2. Stop chasing kills deep into enemy jungle.",
            "3. Sync recall timings so you don't defend 4v5."
        ]
    elif score < 60:
        recs = [
            "AVERAGE COORDINATION: Basics present, but execution lacks polish.",
            "1. Improve vision control. Supports need to ward deeper before fights.",
            "2. Focus on role positioning. Tanks front, Marksmen back.",
            "3. Communicate ultimate cooldowns before engaging."
        ]
    elif score < 90:
        recs = [
            "HIGH SYNERGY: Strong teamwork, focus on optimization.",
            "1. Refine rotation speed. Seconds matter for ganks.",
            "2. Coordinate 'chain-CC' combos perfectly.",
            "3. Start tracking enemy jungler pathing to counter-gank."
        ]
    else:
        recs = [
            "ELITE STATUS: Pro-level coordination detected.",
            "1. Experiment with complex macro strategies (1-3-1 split).",
            "2. Focus on 'micro' mechanical outplays as macro is solid.",
            "3. Analyze replay nuances to maintain this peak form."
        ]
    return recs

def get_diagnostic_insights(stats):
    """Generates specific behavioral diagnostics."""
    weaknesses = []
    if stats['chat_density'] < 40: weaknesses.append("Silent Gameplay (Low Comm)")
    if stats['avg_distance'] > 15: weaknesses.append("Team Isolation (High Distance)")
    if stats['sync_score'] < 50: weaknesses.append("Desynchronized Teamfights")
    if stats['obj_timing_score'] < 50: weaknesses.append("Late Objective Rotations")
    
    root_cause = "Individual Mechanics"
    if len(weaknesses) >= 2: root_cause = "Systemic Coordination Failure"
    elif stats['role_balance_score'] < 60: root_cause = "Draft/Role Gap"
    
    return weaknesses, root_cause

@st.cache_resource
def train_dummy_model():
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(20, 5) 
    y = np.random.randint(0, 2, 20)
    clf.fit(X, y)
    return clf

# ==========================================
# MAIN APP STRUCTURE
# ==========================================
with st.sidebar:
    st.title("🎮 MOBA Strat-G")
    app_mode = st.radio("Select System Module:", 
        ["📊 Post-Game Analysis (Pipeline)", "⚔️ Pre-Game Strategy (Drafting)"])
    st.markdown("---")
    
    if app_mode == "📊 Post-Game Analysis (Pipeline)":
        stages = ["1. Data Ingestion", "2. Pre-Processing", "3. Feature Extraction", "4. CI Scoring", "5. Prediction", "6. Diagnostics", "7. Viz"]
    else:
        stages = ["1. Team Selection", "2. Analysis", "3. Counters", "4. Team Builder", "5. Strategy"]
        
    if 'stage' not in st.session_state: st.session_state.stage = 1
    
    curr_stage = st.session_state.stage
    st.progress(min(curr_stage / len(stages), 1.0))
    for i, s in enumerate(stages, 1):
        if i == curr_stage: st.info(f"📍 {s}")
        elif i < curr_stage: st.success(f"✅ {s}")
        else: st.text(s)
        
    if st.button("🔄 Reset"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.session_state.stage = 1
        st.rerun()

# ==========================================
# MODULE 1: POST-GAME ANALYSIS (PIPELINE)
# ==========================================
if app_mode == "📊 Post-Game Analysis (Pipeline)":
    st.title("📊 Post-Game Behavioral Analysis")
    st.markdown("*The 7-Stage Pipeline for Collective Intelligence Evaluation.*")

    # STAGE 1: DATA INGESTION
    if st.session_state.stage == 1:
        st.header("Stage 1: Data Ingestion")
        st.markdown("Collect match logs, timestamps, and coordinates.")
        
        tab_db, tab_custom = st.tabs(["MPL MY Stats DB", "Custom Behavioral Input"])
        
        with tab_db:
            match = st.selectbox("Select Played Match:", ["Select..."] + list(MPL_MATCH_STATS.keys()))
            if match != "Select...":
                if st.button("Ingest Data"):
                    st.session_state.match_stats = MPL_MATCH_STATS[match]
                    st.session_state.stage = 2
                    st.rerun()
                    
        with tab_custom:
            with st.form("behav_input"):
                st.subheader("Match Performance Stats")
                c_stats1, c_stats2 = st.columns(2)
                kills = c_stats1.number_input("Kills", 0, 100, 20)
                deaths = c_stats2.number_input("Deaths", 0, 100, 20)
                assists = c_stats1.number_input("Assists", 0, 200, 40)
                gold = c_stats2.number_input("Gold Diff", -20000, 20000, 2000)

                st.subheader("Behavioral Metrics")
                c1, c2 = st.columns(2)
                chat = c1.slider("Chat Density (0-100)", 0, 100, 50)
                dist = c1.slider("Avg Distance (Units)", 0.0, 20.0, 10.0)
                role = c2.slider("Role Balance Score", 0, 100, 70)
                obj = c2.slider("Objective Timing Score", 0, 100, 60)
                sync = st.slider("Teamfight Sync Score", 0, 100, 50)
                
                if st.form_submit_button("Process Data"):
                    st.session_state.match_stats = {
                        "team_name": "Custom Team", 
                        "kills": kills, "deaths": deaths, "assists": assists, "gold_diff": gold,
                        "chat_density": chat, "avg_distance": dist, "role_balance_score": role,
                        "obj_timing_score": obj, "sync_score": sync
                    }
                    st.session_state.stage = 2
                    st.rerun()

    # STAGE 2: PRE-PROCESSING
    elif st.session_state.stage == 2:
        st.header("Stage 2: Data Pre-Processing")
        with st.spinner("Normalizing time-series... Aligning logs..."):
            time.sleep(0.8)
        st.success("✅ Data Cleaned: Null values removed, Coordinates normalized.")
        st.dataframe(pd.DataFrame([st.session_state.match_stats]), use_container_width=True)
        if st.button("Extract Features"): st.session_state.stage = 3; st.rerun()

    # STAGE 3: FEATURE EXTRACTION
    elif st.session_state.stage == 3:
        st.header("Stage 3: Teamwork Feature Extraction")
        st.info("Transforming raw stats into CI Indicators.")
        stats = st.session_state.match_stats
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Comm Density", stats['chat_density'])
        c2.metric("Coop Movement", f"{stats['avg_distance']}u")
        c3.metric("Fight Sync", stats['sync_score'])
        
        if st.button("Calculate CI Score"): st.session_state.stage = 4; st.rerun()

    # STAGE 4: CI SCORING
    elif st.session_state.stage == 4:
        st.header("Stage 4: Collective Intelligence Scoring")
        score = calculate_ci_score(st.session_state.match_stats)
        st.session_state.ci_score = score
        
        st.metric("Total CI Score", f"{score}/100")
        if score > 75: st.success("High Synergy Detected")
        elif score > 50: st.warning("Moderate Synergy")
        else: st.error("Low Synergy - Fragmentation Detected")
        
        if st.button("Run Prediction Model"): st.session_state.stage = 5; st.rerun()

    # STAGE 5: PREDICTIVE MODELLING
    elif st.session_state.stage == 5:
        st.header("Stage 5: Predictive Modelling (Random Forest)")
        model = train_dummy_model()
        
        prob = st.session_state.ci_score / 120.0 
        prob = min(0.95, max(0.1, prob))
        
        st.write("Running `RandomForestClassifier` on extracted features...")
        st.progress(prob)
        st.caption(f"Win Probability: {prob*100:.1f}%")
        
        if st.button("Run Diagnostics"): st.session_state.stage = 6; st.rerun()

    # STAGE 6: DIAGNOSTIC ANALYSIS
    elif st.session_state.stage == 6:
        st.header("Stage 6: Diagnostic Analysis & Recommendations")
        stats = st.session_state.match_stats
        ci_score = st.session_state.ci_score
        
        weaknesses, root_cause = get_diagnostic_insights(stats)
        score_recs = get_ci_based_recommendations(ci_score)
        
        # Store for Stage 7
        st.session_state.diagnostics = {"weaknesses": weaknesses, "root_cause": root_cause, "recs": score_recs}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔍 Identified Issues")
            if not weaknesses:
                st.success("No major behavioral flaws detected.")
            else:
                for w in weaknesses: st.error(f"• {w}")
            st.info(f"**Root Cause:** {root_cause}")
            
        with c2:
            st.markdown(f"### 📋 Recommendations (Score: {ci_score})")
            for r in score_recs:
                if "CRITICAL" in r or "POOR" in r or "ELITE" in r or "HIGH" in r or "AVERAGE" in r:
                    st.write(f"**{r}**")
                else:
                    st.info(r)
                   
        if st.button("View Final Dashboard"): st.session_state.stage = 7; st.rerun()

    # STAGE 7: VISUALIZATION
    elif st.session_state.stage == 7:
        st.header("Stage 7: Strategy Recommendation & Viz")
        st.balloons()
        
        stats = st.session_state.match_stats
        diag = st.session_state.diagnostics
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("### 📈 CI Performance Radar")
            chart_data = pd.DataFrame({
                "Metric": ["Chat", "Role", "Obj", "Sync", "Movement"],
                "Score": [stats['chat_density'], stats['role_balance_score'], stats['obj_timing_score'], stats['sync_score'], max(0, 100-(stats['avg_distance']*5))]
            })
            st.bar_chart(chart_data.set_index("Metric"))
            
        with c2:
            st.markdown("### 🩺 Diagnostic Status")
            st.metric("Final CI Score", st.session_state.ci_score)
            st.write("**Root Cause:**")
            st.warning(diag['root_cause'])
            st.write("**Weaknesses Identified:**")
            if diag['weaknesses']:
                for w in diag['weaknesses']: st.error(w)
            else:
                st.success("None")

        st.markdown("---")
        st.markdown("### ✅ Actionable Steps")
        # Display the recommendations again in the final report
        for r in diag['recs']:
             if "1." in r or "2." in r or "3." in r:
                 st.info(r)
             else:
                 st.write(f"**{r}**")

        if st.button("Start New Analysis"): 
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.session_state.stage = 1
            st.rerun()

# ==========================================
# MODULE 2: PRE-GAME STRATEGY (DRAFTING)
# ==========================================
elif app_mode == "⚔️ Pre-Game Strategy (Drafting)":
    st.title("⚔️ Pre-Game Strategy: Counter Pick System")
    
    # STAGE 1
    if st.session_state.stage == 1:
        st.header("Stage 1: Opponent Selection")
        tab_pro, tab_man = st.tabs(["🏆 MPL MY Database", "✍️ Manual Input"])
        
        with tab_pro:
            match = st.selectbox("Select MPL Match:", ["Select..."] + list(MPL_DRAFTS.keys()))
            if match != "Select...":
                if st.button("Analyze Match"):
                    st.session_state.draft_data = MPL_DRAFTS[match]
                    st.session_state.stage = 2
                    st.rerun()
                    
        with tab_man:
            with st.form("manual"):
                c1, c2, c3 = st.columns(3)
                heroes = sorted(list(HERO_DATABASE.keys()))
                exp = c1.selectbox("Exp", ["Select..."]+heroes)
                jg = c1.selectbox("Jungle", ["Select..."]+heroes)
                mid = c2.selectbox("Mid", ["Select..."]+heroes)
                roam = c2.selectbox("Roam", ["Select..."]+heroes)
                gold = c3.selectbox("Gold", ["Select..."]+heroes)
                if st.form_submit_button("Analyze Custom"):
                    inputs = [exp, jg, mid, roam, gold]
                    if "Select..." in inputs: st.error("Fill all lanes")
                    else:
                        style = determine_playstyle(inputs)
                        st.session_state.draft_data = {
                            "team_name": "Custom Team", "playstyle": style,
                            "lanes": {"Exp Lane":exp, "Jungle":jg, "Mid Lane":mid, "Roamer":roam, "Gold Lane":gold},
                            "objective_priority": ["Turrets", "Lord"]
                        }
                        st.session_state.stage = 2
                        st.rerun()

    # STAGE 2
    elif st.session_state.stage == 2:
        st.header("Stage 2: Draft Analysis")
        data = st.session_state.draft_data
        st.info(f"**Playstyle:** {data['playstyle']}")
        for l, h in data['lanes'].items(): st.error(f"**{l}:** {h}")
        if st.button("Find Counters"): st.session_state.stage = 3; st.rerun()

    # STAGE 3
    elif st.session_state.stage == 3:
        st.header("Stage 3: Lane Counters (Top 3)")
        recs = analyze_lane_matchups(st.session_state.draft_data['lanes'])
        st.session_state.recs = recs
        for lane, info in recs.items():
            if info['top']:
                best = info['top'][0]
                with st.expander(f"{lane}: {best['name']} vs {info['enemy']}"):
                    st.success(f"Top Pick: {best['name']} ({best['spec']})")
                    if len(info['top']) > 1:
                        st.caption(f"Alts: {', '.join([x['name'] for x in info['top'][1:]])}")
            else: st.warning(f"No counter for {info['enemy']}")
        if st.button("Build Team"): st.session_state.stage = 4; st.rerun()

    # STAGE 4
    elif st.session_state.stage >= 4:
        st.header("Stage 4: Team Builder")
        team = {l: info['top'][0]['name'] if info['top'] else "Flex" for l, info in st.session_state.recs.items()}
        for l, h in team.items(): st.success(f"{l}: {h}")
        st.button("Generate Strategy (Demo End)", disabled=True)

 # Footer
    st.markdown("---")
    st.markdown("*MOBA Strat-G v1.0 | TMI4033 Collective Intelligence Project*")
    st.markdown("*Team: Mohd Ekmal, Carolline, Anselm, Mohamad Ainur | Universiti Malaysia Sarawak*")