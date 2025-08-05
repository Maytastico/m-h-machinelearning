# Wie ist eine Windturbine aufgebaut?

Beim näheren Blick auf eine Windturbine scheinen die meisten Features sich auf die direkte Umgebung, wenn nicht gar, den Generator selbst einzubeziehen. (https://www.weltderphysik.de/gebiet/technik/energie/windenergie/technik-der-windkraft/)

```
Allgemeine Informationen:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 53093 entries, 0 to 53092
Data columns (total 12 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   turbine_id              53093 non-null  object 
 1   timestamp               53093 non-null  object 
 2   turbine_type            53093 non-null  object 
 3   maintenance_team        53093 non-null  object 
 4   wind_speed_ms           53093 non-null  float64
 5   wind_direction          53093 non-null  object 
 6   power_output_kw         53093 non-null  float64
 7   vibration_mm_s          52032 non-null  float64
 8   temperature_c           52563 non-null  float64
 9   hydraulic_pressure_bar  53093 non-null  float64
 10  days_since_maintenance  53093 non-null  int64  
 11  failure_risk_30d        53093 non-null  int64  
dtypes: float64(5), int64(2), object(5)
memory usage: 4.9+ MB
```

# Kernausfallkriterien einer Windturbine

Nach kurzer Recherche scheinen die folgenden Ausfallkriterien eine größere Rolle zu spielen. 

Zunächst sind Vibrationen auf Langzeit mit Schäden im Antriebsstrang korrelliert. (https://www.amazonfilters.com/de/fallstudie/ausfallrate-von-windturbinengetrieben-verringern)

Außerdem scheint ein großes Ausfallkriterium der Hydraulikdruck zu sein (https://www.amazonfilters.com/de/fallstudie/ausfallrate-von-windturbinengetrieben-verringern). Ein anhaltender hoher Druck schädigt das Getriebe, womit eine Ausfall der Turbine unvermeidlich ist.

Des Weiteren spiegelt eine hohe Temperatur Probleme (bestehend oder anbahnend) wieder, da diese vorallem mit Reibungen (Abnutzung) korrellieren. (https://www.amazonfilters.com/de/fallstudie/ausfallrate-von-windturbinengetrieben-verringern)

## Physische und Technische Zusammenhänge
* Die Vibrationen korrellieren mit der Temperatur, da eine hohe Temperatur zu einer höheren Reibung führt, was wiederum zu höheren Vibrationen führt.
* Ist der Hydraulische Druck zu hoch, so kann dies zu einer Überhitzung führen, da die Hydraulikflüssigkeit nicht mehr richtig zirkulieren kann. Dies führt wiederum zu höheren Temperaturen und Vibrationen.
* Ist die letzte Wartung zu lange her, so kann dies zu höheren Vibrationen und Temperaturen führen, da die Turbine nicht mehr richtig gewartet wird und somit anfälliger für Schäden ist.

## Predictive Failures
Die folgenden Features scheinen für die Vorhersage von Ausfällen am relevantesten zu sein:
* vibration_mm_s
* temperature_c
* hydraulic_pressure_bar

# Feature Transformation
* Der Turbinen Typ könnte in eine One-Hot-Encoding Variable umgewandelt werden, um die verschiedenen Typen zu unterscheiden.
* Die Windrichtung könnte über Label Encoding in eine numerische Variable umgewandelt werden.
* Die Zeitstempel könnten in eine numerische Variable umgewandelt werden.

# Interaktions Feature
Bei Interaktions Features handelt sich um Features, die unterschiedliche Metriken kombinieren um neue Erkenntnisste über
das Verhalten der Windturbine zu gewinnen. Diese könnten wie folgt aussehen:

1. Vibrationsbelastung im Zeitverlauf
`vibreation_per_day = vibration_mm_s / (days_since_maintenance+1)`

Dieses Feature betrachtet wenn bei einer Wartung die Vibrationen
zu hoch sind, könnte bei der Wartung etwas nicht beachtet worden sein.
2. Hohe Vibration nach längerer Zeit
`vibration_accumulated = vibration_mm_s * (days_since_maintenance + 1)`

Steigt die Vibration des System mit längerer Zeit an, so könnte dies auf ein Problem hindeuten.
3. Thermomechanische Belastung
`df["vibration_temp_interaction"] = df["vibration_mm_s"] * df["temperature_c"]`

Mechanische Vibrationen in Verbindung mit hohen Temperaturen führen zu beschleunigtem Verschleiß durch Materialermüdung oder Ölabbau.
4. Wind-Leistungs-Effizienz
`df["wind_power_efficiency"] = df["power_output_kw"] / (df["wind_speed_ms"]**3 + 1e-6)`

Die theoretische Windenergie wächst mit der dritten Potenz der Windgeschwindigkeit. Abweichungen deuten auf Ineffizienzen im System (z.B. Pitchfehler, Leistungsverlust) hin.
5. Hydraulikdruck-Effizienz
`df["hydraulic_power_interaction"] = df["hydraulic_pressure_bar"] * df["power_output_kw"]`

Ein hoher hydraulischer Druck kombiniert mit hoher Leistung könnte auf ineffiziente hydraulische Regelung oder auf Energieverluste hindeuten.
6. Wartungsbedingter Leistungsverlust
`df["performance_decay"] = df["power_output_kw"] / (df["days_since_maintenance"] + 1)`

Sinkt die Leistung im Verhältnis zur Zeit seit letzter Wartung, könnte das auf schleichenden Leistungsverlust durch Verschleiß oder Verunreinigung hinweisen.
7. Dynamische Belastung durch Wind
`df["dynamic_stress"] = df["vibration_mm_s"] * df["wind_speed_ms"]`

Bei starker Vibration und gleichzeitigem hohen Winddruck ist die mechanische Belastung auf Struktur und Rotorflügel besonders hoch.
8. Überhitzungsgefahr bei Windlast
`df["wind_heat_index"] = df["temperature_c"] * df["wind_speed_ms"]`

Bei starkem Wind sollte die Turbine eher kühlen. Eine Kombination aus hoher Temperatur und hoher Windgeschwindigkeit kann auf abnormale Reibung oder ineffiziente Kühlung hindeuten.

# Zeitbassierte Features

1. Rollender Mittelwert der Vibrationen 24h

```python
df["vibration_rollmean_24h"] = (
    df.groupby("turbine_id")["vibration_mm_s"]
    .rolling(window=24, min_periods=3)
    .mean()
    .reset_index(level=0, drop=True)
)
```
Ein zunehmender Mittelwert kann auf schleichenden Verschleiß hinweisen.

2. Rollende Standardabweichung der Temperatur 6h

```python
df["temp_rollstd_6h"] = (
    df.groupby("turbine_id")["temperature_c"]
    .rolling(window=6, min_periods=3)
    .std()
    .reset_index(level=0, drop=True)
)
```
Stark schwankende Temperaturen können auf instabile Betriebsbedingungen oder ein defektes Kühlsystem hinweisen.

3. Rollendes Minimum des hydraulischen Drucks 48h

```python
df["hydraulic_pressure_min_48h"] = (
    df.groupby("turbine_id")["hydraulic_pressure_bar"]
    .rolling(window=48, min_periods=3)
    .min()
    .reset_index(level=0, drop=True)
)
```
Ein wiederkehrend niedriger Minimalwert über 2 Tage kann ein Indikator für Leckagen oder Pumpenprobleme sein.

Trend (Differenz) der Leistung 12h

```python
df["power_output_diff_12h"] = (
    df.groupby("turbine_id")["power_output_kw"]
    .diff(periods=12)
)
```
Ein signifikanter Abfall der Leistung über 12 Stunden kann ein frühes Ausfallsignal sein.

# Wichtigste Features

Nach RandomForest
Top Features (Random Forest):
                       feature  importance
16  hydraulic_pressure_min_48h    0.124801
9        wind_power_efficiency    0.113131
14      vibration_rollmean_24h    0.093215
5       days_since_maintenance    0.073894
19                       month    0.070130

Nach RFE
Top Features (RFE): ['vibration_per_day', 'wind_power_efficiency', 'turbine_type_Enercon E-126', 'turbine_type_GE 1.5MW', 'maintenance_team_Team_A']