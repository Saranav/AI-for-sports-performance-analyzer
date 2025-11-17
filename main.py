import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from math import pi

# ------------------------------
# Step 1: Dataset for Multiple Sports
# ------------------------------
np.random.seed(42)
sports = ['Discus Throw', 'Basketball', 'Cricket']
data = []
for sport in sports:
    for i in range(50):
        speed = np.random.randint(5, 30)
        stamina = np.random.randint(50, 100)
        accuracy = np.random.randint(40, 100)
        workload = np.random.randint(20, 100)
        injury_risk = np.random.choice([0,1])
        data.append([sport, speed, stamina, accuracy, workload, injury_risk])

columns = ['Sport', 'Speed', 'Stamina', 'Accuracy', 'Workload', 'Injury_Risk']
df = pd.DataFrame(data, columns=columns)

# ------------------------------
# Step 2: Train/Test Split for ML
# ------------------------------
X = df[['Speed', 'Stamina', 'Accuracy', 'Workload']]
y = df['Injury_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 3: Train Random Forest Model
# ------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Step 4: Evaluate Model
# ------------------------------
y_pred = model.predict(X_test)
print("\n=== Model Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------------------
# Step 5: Analysis Functions
# ------------------------------
def analyze_player(player_data, sport_name='Unknown', player_name='Player'):
    prediction = model.predict([player_data])[0]
    risk = "High Injury Risk" if prediction == 1 else "Low Injury Risk"

    suggestions = []
    if player_data[1] < 60:
        suggestions.append("Increase stamina through endurance training.")
    if player_data[2] < 60:
        suggestions.append("Improve accuracy with focused drills.")
    if player_data[3] > 80:
        suggestions.append("Reduce workload to prevent burnout.")

    stats_labels = ['Speed','Stamina','Accuracy','Workload']
    best_attr = stats_labels[np.argmax(player_data)]
    weak_attr = stats_labels[np.argmin(player_data)]

    print(f"\n=== Analysis for {player_name} ({sport_name}) ===")
    print("Injury Risk Status:", risk)
    print("Best Attribute:", best_attr)
    print("Weakest Attribute:", weak_attr)
    print("Training Suggestions:")
    for tip in suggestions:
        print("-", tip)

    sport_avg = df[df['Sport']==sport_name][['Speed','Stamina','Accuracy','Workload']].mean()
    print("Benchmark Comparison (avg stats for sport):")
    for i, label in enumerate(stats_labels):
        status = "Above Average" if player_data[i] >= sport_avg[label] else "Below Average"
        print(f"{label}: {player_data[i]} ({status}, Avg: {sport_avg[label]:.2f})")

    if player_data[3] > 90 or player_data[1] < 50:
        print("⚠ Warning: Player may be at critical injury risk!")

    training_plan = []
    if player_data[1] < 70:
        training_plan.append("Endurance drills 3x/week")
    if player_data[2] < 70:
        training_plan.append("Accuracy drills 3x/week")
    if player_data[3] > 75:
        training_plan.append("Reduce workload, include rest days")
    print("Suggested Weekly Training Plan:")
    for plan in training_plan:
        print("-", plan)

    return risk, suggestions, training_plan

# Radar chart
def radar_chart(player_name, player_data):
    labels = ['Speed','Stamina','Accuracy','Workload']
    stats = player_data + player_data[:1]
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels)
    ax.plot(angles, stats, linewidth=2, linestyle='solid', label=player_name)
    ax.fill(angles, stats, 'b', alpha=0.1)
    plt.title(f'{player_name} Stats Radar Chart')
    plt.legend()
    plt.show()

# Plot trend
def plot_trend(player_name, speeds, staminas, accuracies):
    plt.figure(figsize=(10,5))
    plt.plot(speeds, label='Speed', marker='o')
    plt.plot(staminas, label='Stamina', marker='o')
    plt.plot(accuracies, label='Accuracy', marker='o')
    plt.xlabel('Weeks')
    plt.ylabel('Metrics')
    plt.title(f'Performance Trend for {player_name}')
    plt.legend()
    plt.show()

# ------------------------------
# Step 6: User-defined Players
# ------------------------------
num_players = int(input("Enter number of players: "))
user_players = []

for i in range(num_players):
    name = input(f"\nEnter name for Player {i+1}: ")
    sport = input(f"Enter sport for {name} (Discus Throw/Basketball/Cricket): ")
    # Generate random stats for the player
    speed = np.random.randint(20,30)
    stamina = np.random.randint(50,90)
    accuracy = np.random.randint(50,90)
    workload = np.random.randint(40,90)
    stats = [speed, stamina, accuracy, workload]
    user_players.append({'name':name,'sport':sport,'stats':stats})

# Analyze and visualize
for player in user_players:
    analyze_player(player['stats'], sport_name=player['sport'], player_name=player['name'])
    radar_chart(player['name'], player['stats'])
    # Simulate weekly trends for 5 weeks
    weekly_speeds = [player['stats'][0] + np.random.randint(-2,3) for _ in range(5)]
    weekly_staminas = [player['stats'][1] + np.random.randint(-2,3) for _ in range(5)]
    weekly_accuracies = [player['stats'][2] + np.random.randint(-2,3) for _ in range(5)]
    plot_trend(player['name'], weekly_speeds, weekly_staminas, weekly_accuracies)

print("\nAll player analyses complete ✅")
