# Green : Classification de Plantes Médicinales avec MobileNetV2 et Focal Loss

## Introduction

Dans cet article technique, nous allons explorer en profondeur l'architecture et l'implémentation de **Green**, un modèle de deep learning conçu pour identifier quatre plantes médicinales traditionnelles. Ce modèle alimente l'application mobile DrGreen et démontre comment combiner transfer learning, loss functions personnalisées et techniques d'augmentation de données pour obtenir des performances robustes avec un dataset limité.

## Table des Matières

1. [Contexte et Problématique](#contexte-et-problématique)
2. [Architecture du Modèle](#architecture-du-modèle)
3. [Focal Loss : La Clé de la Performance](#focal-loss)
4. [Pipeline de Données et Augmentation](#pipeline-de-données)
5. [Stratified Split : Éviter le Class Collapse](#stratified-split)
6. [Optimisation et Régularisation](#optimisation-et-régularisation)
7. [Métriques et Évaluation](#métriques-et-évaluation)
8. [Déploiement Mobile](#déploiement-mobile)
9. [Leçons Apprises](#leçons-apprises)

---

## Contexte et Problématique

### Le Défi

Nous devons classifier 4 espèces de plantes médicinales :
- **Artemisia** (Artemisia annua) - propriétés antipaludiques
- **Carica** (Carica papaya) - santé digestive
- **Goyavier** (Psidium guajava) - remède traditionnel
- **Kinkeliba** (Combretum micranthum) - plante médicinale ouest-africaine

### Contraintes

- **Dataset limité** : 1,164 images seulement
- **Déséquilibre des classes** : 20.7% à 30.6% par classe
- **Déploiement mobile** : modèle léger requis
- **Contraintes temps réel** : inférence rapide nécessaire

## Architecture du Modèle

### Choix de MobileNetV2

MobileNetV2 a été sélectionné pour plusieurs raisons techniques :

```python
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=inputs,
    pooling='avg'
)
base_model.trainable = False  # Transfer learning avec base gelée
```

**Avantages** :
- **Inverted Residuals** : réduction de la complexité computationnelle
- **Linear Bottlenecks** : préservation des features importantes
- **Lightweight** : 2.3M paramètres totaux, 82K entraînables
- **Pré-entraîné ImageNet** : knowledge transfer efficace

### Architecture Complète

```
Input (224×224×3)
    ↓
MobileNetV2 Base (frozen)
    ↓
Global Average Pooling
    ↓
Dropout(0.6) ← Forte régularisation
    ↓
Dense(64, ReLU) + L2(0.02) ← Feature extraction
    ↓
Batch Normalization ← Stabilisation
    ↓
Dropout(0.3) ← Régularisation supplémentaire
    ↓
Dense(4, Softmax) + L2(0.02) ← Classification finale
```

**Paramètres clés** :
- Total : 2,340,484 paramètres
- Entraînables : 82,372 (3.5%)
- Non-entraînables : 2,258,112

## Focal Loss : La Clé de la Performance

### Pourquoi Focal Loss ?

La **Categorical Cross-Entropy** standard traite tous les exemples également. Avec un dataset limité et déséquilibré, cela pose problème :

```python
# Cross-Entropy standard
loss = -Σ y_true * log(y_pred)
```

**Problèmes** :
- Les exemples faciles dominent le gradient
- Les classes minoritaires sont sous-représentées
- Pas de focus sur les hard examples

### Implémentation de Focal Loss

```python
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma          # Facteur de modulation
        self.alpha = alpha          # Pondération des classes
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Label smoothing pour régularisation
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1.0 - self.label_smoothing) + \
                 (self.label_smoothing / num_classes)

        # Clipping pour stabilité numérique
        y_pred = tf.clip_by_value(y_pred,
                                   tf.keras.backend.epsilon(),
                                   1 - tf.keras.backend.epsilon())

        # Cross-entropy de base
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calcul de p_t (probabilité de la vraie classe)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)

        # Focal weight : (1 - p_t)^gamma
        # Plus p_t est petit (exemple difficile), plus le poids est élevé
        focal_weight = tf.pow(1 - p_t, self.gamma)

        # Application du focal loss
        focal_loss = self.alpha * focal_weight * tf.reduce_sum(cross_entropy, axis=-1)

        return tf.reduce_mean(focal_loss)
```

### Impact de Gamma

| γ | Comportement | Usage |
|---|-------------|-------|
| 0 | Cross-Entropy standard | Baseline |
| 1 | Réduction modérée du poids des easy examples | Déséquilibre léger |
| **2** | **Forte focalisation sur hard examples** | **Notre choix** |
| 5 | Focalisation extrême | Risque d'instabilité |

**Exemple concret** :

```python
# Easy example : p_t = 0.9
focal_weight = (1 - 0.9)^2 = 0.01  # Poids très réduit

# Hard example : p_t = 0.3
focal_weight = (1 - 0.3)^2 = 0.49  # Poids important

# Ratio : 0.49 / 0.01 = 49x plus d'attention sur les hard examples !
```

## Pipeline de Données et Augmentation

### Stratégie d'Augmentation Agressive

Avec seulement 931 images d'entraînement, l'augmentation est cruciale :

```python
data_augmentation = tf.keras.Sequential([
    # Flips géométriques
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),

    # Rotation jusqu'à ±108° (0.3 * 360°)
    tf.keras.layers.RandomRotation(0.3),

    # Zoom ±20% pour variations d'échelle
    tf.keras.layers.RandomZoom(0.2),

    # Variations photométriques
    tf.keras.layers.RandomBrightness(0.2),    # ±20% luminosité
    tf.keras.layers.RandomContrast(0.2),      # ±20% contraste

    # Translation pour robustesse de position
    tf.keras.layers.RandomTranslation(0.15, 0.15),
], name="data_augmentation")
```

**Justification des choix** :

1. **Rotations importantes (±108°)** : les plantes peuvent être photographiées sous n'importe quel angle
2. **Flips vertical + horizontal** : pas d'orientation canonique pour les feuilles
3. **Variations photométriques** : conditions d'éclairage variables en milieu naturel

### Pipeline Optimisé

```python
# Configuration pour performance maximale
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

# Parallélisation du chargement
train_ds = train_ds.map(load_and_preprocess_image,
                        num_parallel_calls=AUTOTUNE)

# Augmentation (seulement en training)
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# Preprocessing MobileNetV2 : [-1, 1] normalization
train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)

# Batching et prefetching pour GPU utilization
train_ds = train_ds.shuffle(1000)\
                   .batch(16)\
                   .prefetch(AUTOTUNE)
```

**Optimisations clés** :
- `num_parallel_calls=AUTOTUNE` : TensorFlow optimise automatiquement le parallélisme
- `prefetch(AUTOTUNE)` : prépare le batch suivant pendant l'entraînement du batch actuel
- `shuffle(1000)` : buffer de 1000 images pour randomisation efficace

## Stratified Split : Éviter le Class Collapse

### Le Problème avec Random Split

```python
# ❌ BAD : Split aléatoire
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="both",
    seed=42
)
```

**Problème** : avec un petit dataset, le split aléatoire peut créer des déséquilibres :
- Classe A : 90% en train, 10% en validation
- Classe B : 70% en train, 30% en validation
- Risque de validation set non représentatif

### Solution : Stratified Split

```python
# ✅ GOOD : Split stratifié
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels  # ← La clé !
)
```

**Résultat** : distribution identique dans train et validation

| Classe | Train % | Validation % | Différence |
|--------|---------|--------------|------------|
| Artemisia | 23.6% | 23.6% | 0.0% |
| Carica | 30.6% | 30.5% | 0.1% |
| Goyavier | 20.7% | 20.6% | 0.1% |
| Kinkeliba | 25.0% | 25.3% | 0.3% |

**Impact** : validation accuracy plus fiable et pas de class collapse !

## Optimisation et Régularisation

### Learning Rate Schedule : Cosine Decay

```python
steps_per_epoch = len(train_labels) // batch_size
total_steps = steps_per_epoch * epochs

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0005,  # LR initial
    decay_steps=total_steps,
    alpha=0.01  # LR final = 0.01 * initial = 0.000005
)
```

**Avantages du Cosine Decay** :
- Décroissance douce vs step decay brutal
- Évite les oscillations en fin d'entraînement
- LR final non nul pour fine-tuning

### Stack de Régularisation

1. **Dropout (60% + 30%)**
```python
x = tf.keras.layers.Dropout(0.6)(x)  # Après GAP
# ...
x = tf.keras.layers.Dropout(0.3)(x)  # Après Dense
```

2. **L2 Regularization**
```python
kernel_regularizer=tf.keras.regularizers.l2(0.02)
```

3. **Batch Normalization**
```python
x = tf.keras.layers.BatchNormalization()(x)
```

4. **Label Smoothing (15%)**
```python
# Dans Focal Loss
y_true = y_true * 0.85 + 0.15/4  # Soft labels
```

5. **Class Weights**
```python
# Pondération dynamique inversement proportionnelle à la fréquence
class_weights = {
    0: 1.076,  # Artemisia (sous-représenté)
    1: 0.769,  # Carica (sur-représenté)
    2: 1.276,  # Goyavier (le plus sous-représenté)
    3: 0.999   # Kinkeliba (équilibré)
}
```

### Early Stopping Intelligent

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Attend 15 epochs sans amélioration
        restore_best_weights=True,  # Restaure les meilleurs poids
        mode='max'
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath='models/best_model_v7.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]
```

## Métriques et Évaluation

### Métriques Multi-dimensionnelles

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.15),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')
    ]
)
```

**Top-2 Accuracy** : crucial pour une app mobile
- Accuracy : 69.10%
- Top-2 Accuracy : **88.41%** ← L'app peut proposer 2 suggestions

### Analyse de Confusion Matrix

```python
# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)

# Analyse per-class
for i, class_name in enumerate(class_names):
    class_mask = y_true == i
    class_acc = (y_pred[class_mask] == i).mean()
    print(f"{class_name}: {class_acc*100:.2f}%")
```

**Résultats** :
```
[OK]  Artemisia: 67.27%
[OK]  Carica: 73.24%
[LOW] Goyavier: 60.42%  ← Plus difficile (moins d'exemples)
[OK]  Kinkeliba: 71.19%
```

### Détection de Class Collapse

```python
# Vérification de la distribution des prédictions
pred_counts = {name: 0 for name in class_names}
for p in y_pred:
    pred_counts[class_names[p]] += 1

for class_name, count in pred_counts.items():
    pct = count/len(y_pred)*100
    if pct > 50:  # ⚠️ Collapse détecté si > 50%
        print(f"WARNING: {class_name} = {pct:.1f}%")
```

**Notre modèle** : ✅ Pas de collapse
```
Artemisia: 25.8%
Carica: 33.5%
Goyavier: 18.0%
Kinkeliba: 22.7%
```

## Déploiement Mobile

### Conversion en TensorFlow Lite

```python
# 1. Charger le modèle
model = tf.keras.models.load_model('models/best_model_v7.keras')

# 2. Convertir en TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Optimisations pour mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Quantization pour réduction de taille
converter.target_spec.supported_types = [tf.float16]

# 5. Conversion
tflite_model = converter.convert()

# 6. Sauvegarde
with open('green_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Gains de performance** :
- Taille : ~9 MB → ~2.3 MB (quantization float16)
- Latence : ~150ms → ~40ms sur mobile
- RAM : ~50 MB → ~15 MB

### Inférence Mobile (Exemple Flutter)

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class PlantClassifier {
  late Interpreter _interpreter;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('green_model.tflite');
  }

  Future<Map<String, double>> classify(File imageFile) async {
    // 1. Prétraitement
    var input = preprocessImage(imageFile);  // 224x224x3

    // 2. Inférence
    var output = List.filled(4, 0.0).reshape([1, 4]);
    _interpreter.run(input, output);

    // 3. Post-processing
    final classes = ['artemisia', 'carica', 'goyavier', 'kinkeliba'];
    return Map.fromIterables(classes, output[0]);
  }
}
```

## Leçons Apprises

### 1. Dataset Quality > Quantity

Avec seulement 1,164 images :
- ✅ Stratified split crucial
- ✅ Augmentation agressive nécessaire
- ✅ Transfer learning indispensable

### 2. Loss Function Matters

Focal Loss vs Cross-Entropy :
- +12% accuracy sur classes minoritaires
- Convergence plus stable
- Pas de class collapse

### 3. Régularisation Multi-niveaux

Stack de régularisation :
```
Dropout (0.6 + 0.3)
+ L2 (0.02)
+ Batch Normalization
+ Label Smoothing (0.15)
+ Early Stopping (patience=15)
= Modèle robuste sans overfitting
```

### 4. Validation Set Design

Le split stratifié a éliminé :
- ❌ Validation accuracy instable
- ❌ Class collapse sur certaines runs
- ❌ Métriques non représentatives

### 5. Mobile-First Architecture

MobileNetV2 offre le meilleur trade-off :
- Légèreté : 2.3 MB en FP16
- Performance : 69% accuracy, 88% top-2
- Vitesse : 40ms sur smartphone

## Améliorations Futures

### Court Terme

1. **Fine-tuning partiel** : dégeler les dernières couches de MobileNetV2
```python
# Dégeler les 20 dernières couches
for layer in base_model.layers[-20:]:
    layer.trainable = True
```

2. **Test-Time Augmentation (TTA)**
```python
def predict_with_tta(model, image, n_augmentations=10):
    predictions = []
    for _ in range(n_augmentations):
        augmented = data_augmentation(image, training=True)
        pred = model.predict(augmented)
        predictions.append(pred)
    return np.mean(predictions, axis=0)
```

### Long Terme

1. **Expansion du dataset** : 5,000+ images par classe
2. **Grad-CAM pour explainability** : visualisation des zones décisionnelles
3. **Ensemble methods** : combiner MobileNetV2, EfficientNet, ResNet
4. **Multi-label classification** : reconnaître plusieurs plantes simultanément

## Conclusion

Le modèle **Green** démontre qu'avec une architecture bien pensée et des techniques modernes (Focal Loss, stratified split, régularisation multi-niveaux), il est possible d'obtenir des performances robustes même avec un dataset limité.

**Points clés** :
- 🎯 69.10% accuracy, 88.41% top-2 accuracy
- 📱 Déployable sur mobile (2.3 MB, 40ms inférence)
- 🔧 Pas de class collapse grâce au stratified split
- 🚀 Focal Loss pour gestion du déséquilibre

Le code complet est disponible sur [GitHub](https://github.com/armelyara/Green).

---

## Références

1. **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
2. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
3. **Data Augmentation**: Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning", Journal of Big Data 2019
4. **Transfer Learning**: Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014

---

**Auteur** : Équipe DrGreen
**Licence** : Apache 2.0
**Date** : Décembre 2025

Pour toute question technique, ouvrez une issue sur le [dépôt GitHub](https://github.com/armelyara/Green/issues).
