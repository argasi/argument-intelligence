
# Arguments Intelligence

* This application does two things:

Educational: It explains the concept of "Arguments Intelligence" (using data to reduce cognitive bias).

Functional: It provides a dynamic tool where users can select a life scenario, choose a framework (SWOT, PESTLE, SOAR, etc.), and generate a structured analysis.

### Key Features Explained:
- **🔍 Dynamic Framework Engine: The JavaScript object const frameworks holds the logic for all 7 requested methodologies (SWOT, PESTLE, SOAR, TOWS, Porter's, NOISE, Balanced Scorecard). Changing the dropdown instantly reconfigures the DOM.

- **🔍 Educational Context: The "Concept" section explains why these tools help ordinary people, bridging the gap between corporate strategy and daily life.

- **🔍 Simulated AI Logic: The generateAnalysis() function aggregates user data and provides a context-aware "Simulated AI Insight" based on which framework was chosen (e.g., if you choose Porter's Five Forces, it advises on leverage; if you choose SOAR, it advises on aspirations).

Responsive Design: It uses CSS Grid and Flexbox to work on mobile phones and desktops.