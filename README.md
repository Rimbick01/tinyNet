# EfficientNet in Tinygrad: Minimalist Implementation

A clean, educational implementation of EfficientNet using Tinygrad. This project focuses on:

- **Core architecture** without unnecessary complexity  
- **Educational clarity** over production optimization  
- **Pure Tinygrad** without framework hybrids  
- **Minimal dependencies** for easy experimentation  

## 🧩 What's Inside

### Model Components
| Component | Description |
|-----------|-------------|
| **MBConv Blocks** | Mobile inverted bottlenecks with depthwise separable convolutions |
| **Squeeze-and-Excitation** | Channel-wise attention mechanism |
| **Stem/Head Layers** | Standard EfficientNet entry/exit blocks |
| **Compound Scaling** | Uniform width/depth/resolution scaling |

### ✨ Features
- 📐 EfficientNet-B0 to B7 configuration support
- 🖼️ Imagenet-class inference capability
- ⚙️ Weight loading utilities
- 🌱 Simple preprocessing pipeline
- 📦 < 200 lines of core implementation

### TODO
  * Add cuda support
  * Add more example
