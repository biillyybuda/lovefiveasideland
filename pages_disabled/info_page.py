import streamlit as st
from utils.ui_components import page_header


def render_info_page():
    page_header(
        "About Love Five-A-Side App",
        "How ratings, chemistry, and rivalries work",
        center=True,
        divider=True,
    )


    # --------------------------
    # 🏆 MMR
    # --------------------------
    with st.expander("🏆 What is MMR?", expanded=False):
        st.markdown(
            r"""
**MMR (Matchmaking Rating)** is your performance score — it rises when you win and falls when you lose.

It’s based on the **Elo system** idea:
- Beat strong opponents → bigger gain  
- Lose to weaker opponents → bigger drop  

### 🔢 Core idea (Elo-style)
The underlying update follows the Elo structure:

$$
MMR_{new} = MMR_{old} + K \times (R - E)
$$

- **R** = actual result (1 win, 0.5 draw, 0 loss)  
- **E** = expected result from team ratings  

Expected result (classic Elo form):

$$
E = \frac{1}{1 + 10^{((MMR_{opp} - MMR_{team})/400)}}
$$

### ⚖️ Fairness adjustments (what makes your app different)
In this app, the effective **K** isn’t always the same:
- If your team is the underdog, your gains can be boosted  
- If your team is the favourite, your gains can be dampened  
- Very high-rated players can have slightly reduced volatility

So: the Elo structure stays the same, but the “how much should this move?” part is smarter.
            """.strip()
        )

    # --------------------------
    # 🤝 CHEMISTRY
    # --------------------------
    with st.expander("🤝 What is Chemistry?"):
        st.markdown(
            r"""
**Chemistry** measures how well two players perform **together** on the same side.

It rewards:
- playing together often (history)
- winning together
- close games (not blowouts)

### 🔢 Chemistry (simplified, matches the app behaviour)
The app uses:

- **games** = matches together  
- **win%** = wins / games  
- **avg_gd** = average goal difference in those matches  

Closeness factor (close games score higher):

$$
Closeness = \max(0.35,\ 1 - \frac{Avg\_GD}{8})
$$

Depth weight (new duos start lower, established duos get boosted):

$$
DepthWeight = 0.5 + 0.5 \times \min\left(1,\ \frac{\log_{10}(Games+1)}{\log_{10}(10)}\right)
$$

Core chemistry shape:

$$
Chemistry \propto (Games \times Win\% \times Closeness) \times DepthWeight
$$

**Extra rule:** if a duo has **0 wins**, chemistry is halved to keep winless pairs near the bottom.

### 💡 In plain English
- Win a lot together → high chemistry  
- Play tight games together → even higher  
- One-off pairings stay low until there’s enough history  
            """.strip()
        )

    # --------------------------
    # 🔥 RIVALRY INTENSITY
    # --------------------------
    with st.expander("🔥 What is Rivalry Intensity?"):
        st.markdown(
            r"""
**Rivalry Intensity** measures how fierce and competitive two opponents are.

It rewards:
- meeting often
- trading wins (balanced rivalry)
- close scorelines

### 🔢 Intensity (simplified, matches the app behaviour)
- **games** = total meetings  
- **diff** = absolute win% difference between the two (0 = perfectly even)  
- **avg_gd** = average goal difference (lower = closer)

Closeness factor (capped at 5):

$$
Closeness = 1 - \frac{\min(Avg\_GD, 5)}{5}
$$

Depth weight (more history → higher weight):

$$
DepthWeight = \min\left(1,\ \frac{\log_{10}(Games+1)}{\log_{10}(10)}\right)
$$

Core intensity shape:

$$
Intensity \propto (Games \times (1 - Diff) \times Closeness) \times DepthWeight
$$

**Extra rule:** very small sample rivalries (**≤ 2 games**) are dampened so they don’t rank too highly.

### 💡 In plain English
- Play often + trade wins + tight scorelines → “proper rivalry”  
- One-sided or rare matchups stay low  
            """.strip()
        )

