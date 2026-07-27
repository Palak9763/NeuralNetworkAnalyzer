/**
 * components/ExamplesPage.tsx
 *
 * Showcases common neural network architectures with downloadable
 * PyTorch code snippets that users can upload to try the analyzer.
 */

import { useState } from "react";

interface Example {
  id: string;
  name: string;
  category: string;
  categoryColor: string;
  description: string;
  layers: string[];
  params: string;
  inputShape: string;
  useCase: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  difficultyColor: string;
  code: string;
}

const EXAMPLES: Example[] = [
  {
    id: "simple-mlp",
    name: "Simple MLP",
    category: "Feedforward",
    categoryColor: "#4285f4",
    description:
      "A basic multi-layer perceptron with 3 fully connected layers. Great starting point for understanding neural network fundamentals.",
    layers: ["Linear(784, 256)", "ReLU", "Linear(256, 128)", "ReLU", "Linear(128, 10)"],
    params: "~235K",
    inputShape: "784 (28×28 flattened)",
    useCase: "MNIST digit classification",
    difficulty: "Beginner",
    difficultyColor: "#34a853",
    code: `import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

model = SimpleMLP()`,
  },
  {
    id: "mini-cnn",
    name: "Mini CNN",
    category: "CNN",
    categoryColor: "#a855f7",
    description:
      "A compact convolutional neural network with 2 conv blocks followed by a classifier. Ideal for small image classification tasks.",
    layers: ["Conv2d(1,32)", "ReLU", "MaxPool2d", "Conv2d(32,64)", "ReLU", "MaxPool2d", "Linear(9216,128)", "Linear(128,10)"],
    params: "~1.2M",
    inputShape: "1 × 28 × 28",
    useCase: "Image classification (grayscale)",
    difficulty: "Beginner",
    difficultyColor: "#34a853",
    code: `import torch
import torch.nn as nn

class MiniCNN(nn.Module):
    """Compact CNN for grayscale image classification."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = MiniCNN()`,
  },
  {
    id: "mini-resnet",
    name: "MiniResNet",
    category: "ResNet",
    categoryColor: "#ea4335",
    description:
      "A simplified ResNet with BasicBlock residual connections. Demonstrates skip connections and identity shortcuts used in deep networks.",
    layers: ["Conv2d(3,16)", "BatchNorm2d", "ReLU", "BasicBlock×4", "BasicBlock×4", "AdaptiveAvgPool2d", "Linear(64,10)"],
    params: "~16.7K",
    inputShape: "3 × 32 × 32",
    useCase: "CIFAR-10 classification",
    difficulty: "Intermediate",
    difficultyColor: "#fbbc04",
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class MiniResNet(nn.Module):
    """Simplified ResNet for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = MiniResNet()`,
  },
  {
    id: "vgg-lite",
    name: "VGG-Lite",
    category: "VGG",
    categoryColor: "#34a853",
    description:
      "A simplified version of VGGNet with progressive feature maps (64→128→256). Shows the classic pattern of stacking conv layers with increasing depth.",
    layers: ["Conv2d×2(64)", "MaxPool", "Conv2d×2(128)", "MaxPool", "Conv2d×3(256)", "MaxPool", "FC→4096→10"],
    params: "~3.7M",
    inputShape: "3 × 32 × 32",
    useCase: "Image classification",
    difficulty: "Intermediate",
    difficultyColor: "#fbbc04",
    code: `import torch
import torch.nn as nn

class VGGLite(nn.Module):
    """Simplified VGG-style network."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = VGGLite()`,
  },
  {
    id: "autoencoder",
    name: "Convolutional Autoencoder",
    category: "Autoencoder",
    categoryColor: "#ff6d01",
    description:
      "An encoder-decoder architecture that compresses images into a latent space and reconstructs them. Useful for dimensionality reduction and denoising.",
    layers: ["Encoder: Conv2d×3 (↓)", "Bottleneck: 16-dim", "Decoder: ConvTranspose2d×3 (↑)"],
    params: "~50K",
    inputShape: "1 × 28 × 28",
    useCase: "Image reconstruction / denoising",
    difficulty: "Intermediate",
    difficultyColor: "#fbbc04",
    code: `import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for image reconstruction."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = ConvAutoencoder()`,
  },
  {
    id: "transformer-block",
    name: "Mini Transformer",
    category: "Transformer",
    categoryColor: "#e040fb",
    description:
      "A small transformer encoder for sequence classification. Includes multi-head self-attention, layer normalization, and position-wise feedforward layers.",
    layers: ["Embedding", "PositionalEncoding", "TransformerEncoderLayer×2", "Linear"],
    params: "~530K",
    inputShape: "Sequence of tokens (max 512)",
    useCase: "Text classification",
    difficulty: "Advanced",
    difficultyColor: "#ea4335",
    code: `import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MiniTransformer(nn.Module):
    """Small transformer encoder for sequence classification."""
    def __init__(self, vocab_size=10000, d_model=128, nhead=4,
                 num_layers=2, num_classes=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x)

model = MiniTransformer()`,
  },
];

function CodeBlock({ code, filename }: { code: string; filename: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="rounded-xl overflow-hidden border border-white/5">
      <div className="flex items-center justify-between px-4 py-2 bg-[#070910] border-b border-white/5">
        <span className="text-xs text-gray-500 font-mono">{filename}</span>
        <div className="flex gap-2">
          <button
            onClick={handleCopy}
            className="text-xs px-3 py-1 rounded-lg bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition"
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
          <button
            onClick={handleDownload}
            className="text-xs px-3 py-1 rounded-lg bg-accent/20 text-accent hover:bg-accent hover:text-white transition"
          >
            ↓ Download .py
          </button>
        </div>
      </div>
      <pre className="p-4 bg-[#090b11] overflow-auto max-h-[350px] text-sm text-gray-300 font-mono leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export default function ExamplesPage() {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div className="h-full overflow-auto p-6 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-white font-bold text-xl flex items-center gap-2">
          <span className="text-lg">🧪</span> Example Architectures
        </h2>
        <p className="text-gray-400 text-sm mt-1">
          Download any example below, then upload it with <strong>+ Upload Project</strong> to visualize the architecture.
        </p>
      </div>

      {/* Difficulty Legend */}
      <div className="flex gap-4 text-xs">
        {[
          { label: "Beginner", color: "#34a853" },
          { label: "Intermediate", color: "#fbbc04" },
          { label: "Advanced", color: "#ea4335" },
        ].map((d) => (
          <div key={d.label} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
            <span className="text-gray-400">{d.label}</span>
          </div>
        ))}
      </div>

      {/* Examples Grid */}
      <div className="grid gap-4 lg:grid-cols-2">
        {EXAMPLES.map((ex) => {
          const isExpanded = expandedId === ex.id;
          return (
            <div
              key={ex.id}
              className={`rounded-xl border transition-all duration-300 ${
                isExpanded
                  ? "bg-gradient-to-br from-white/[0.03] to-transparent border-accent/30 lg:col-span-2 shadow-lg shadow-accent/5"
                  : "bg-panel border-white/5 hover:border-white/15"
              }`}
            >
              {/* Card Header */}
              <div className="p-5">
                <div className="flex items-start justify-between gap-3 mb-3">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold border border-white/5 shrink-0"
                      style={{ background: `${ex.categoryColor}15`, color: ex.categoryColor }}
                    >
                      {ex.name.charAt(0)}
                    </div>
                    <div>
                      <h3 className="text-white font-semibold">{ex.name}</h3>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span
                          className="text-[10px] font-medium px-2 py-0.5 rounded-full"
                          style={{ background: `${ex.categoryColor}20`, color: ex.categoryColor }}
                        >
                          {ex.category}
                        </span>
                        <span
                          className="text-[10px] font-medium px-2 py-0.5 rounded-full"
                          style={{ background: `${ex.difficultyColor}20`, color: ex.difficultyColor }}
                        >
                          {ex.difficulty}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-gray-400 text-sm leading-relaxed mb-4">{ex.description}</p>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-3 mb-4">
                  {[
                    { label: "Parameters", value: ex.params },
                    { label: "Input", value: ex.inputShape },
                    { label: "Use Case", value: ex.useCase },
                  ].map((s) => (
                    <div key={s.label} className="bg-white/[0.02] rounded-lg p-2.5 border border-white/5">
                      <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">{s.label}</div>
                      <div className="text-xs text-white font-medium truncate" title={s.value}>{s.value}</div>
                    </div>
                  ))}
                </div>

                {/* Layer flow */}
                <div className="flex flex-wrap items-center gap-1.5 mb-4">
                  {ex.layers.map((l, i) => (
                    <span key={i} className="flex items-center gap-1.5">
                      {i > 0 && <span className="text-gray-600 text-xs">→</span>}
                      <span className="text-[11px] px-2 py-1 rounded-md bg-white/5 text-gray-300 font-mono">{l}</span>
                    </span>
                  ))}
                </div>

                {/* Toggle Button */}
                <button
                  onClick={() => setExpandedId(isExpanded ? null : ex.id)}
                  className="text-xs px-4 py-2 rounded-lg bg-accent/15 text-accent hover:bg-accent hover:text-white transition-all duration-200 font-medium"
                >
                  {isExpanded ? "Hide Code ↑" : "View Code & Download ↓"}
                </button>
              </div>

              {/* Expanded Code Section */}
              {isExpanded && (
                <div className="px-5 pb-5">
                  <CodeBlock
                    code={ex.code}
                    filename={`${ex.id.replace(/-/g, "_")}.py`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
