import { useState } from "react";

function SectionHeader({ icon, title }: { icon: string; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-lg">{icon}</span>
      <h3 className="text-white font-semibold text-sm uppercase tracking-wider">{title}</h3>
    </div>
  );
}

const FAQ_ITEMS = [
  {
    q: "What file formats are supported for upload?",
    a: "Currently, the analyzer supports PyTorch model files (.py, .pt, .pth) and traced models. Upload a Python file containing your model class or a traced/scripted model checkpoint.",
  },
  {
    q: "How does the model parsing work?",
    a: "The backend uses PyTorch's tracing mechanism to execute a forward pass with dummy inputs, capturing the computation graph. It then converts this into a universal graph format with nodes, edges, and metadata.",
  },
  {
    q: "What does the confidence level mean?",
    a: "'Traced' means the model was successfully traced with full accuracy. 'Static' means the graph was parsed from static analysis without execution. 'Partial' means some layers couldn't be fully resolved.",
  },
  {
    q: "Can I export the architecture diagram?",
    a: "Yes! Use the Export button in the top bar to download the visualization as PNG, SVG, or JSON. You can also use the Share button to generate a shareable link.",
  },
  {
    q: "Why are some layers showing unknown shapes?",
    a: "Shape inference depends on successful tracing. If the model has dynamic shapes or conditional logic, some output shapes may not be resolved. Try providing explicit input dimensions.",
  },
  {
    q: "How do I interpret the layer type colors?",
    a: "Blue = Convolution, Yellow = Normalization, Green = Activation, Purple = Pooling, Red = Fully Connected, Gray = Other/Utility layers. These colors are consistent across the visualizer and dashboard.",
  },
];

export default function HelpPage() {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  return (
    <div className="h-full overflow-auto p-6 space-y-6">
      <div>
        <h2 className="text-white font-bold text-xl">Help & Documentation</h2>
        <p className="text-gray-400 text-sm mt-1">Learn how to use the Neural Network Analyzer effectively.</p>
      </div>

      <div className="grid gap-5">
        {/* FAQ */}
        <div className="bg-panel rounded-xl p-5 border border-white/5">
          <SectionHeader icon="❓" title="Frequently Asked Questions" />
          <div className="space-y-2">
            {FAQ_ITEMS.map((item, i) => (
              <div key={i} className="rounded-lg border border-white/5 overflow-hidden">
                <button onClick={() => setOpenFaq(openFaq === i ? null : i)}
                  className="w-full flex items-center justify-between p-3 text-left hover:bg-white/[0.02] transition">
                  <span className="text-sm text-white font-medium pr-4">{item.q}</span>
                  <span className={`text-gray-500 shrink-0 transition-transform ${openFaq === i ? "rotate-180" : ""}`}>▼</span>
                </button>
                {openFaq === i && (
                  <div className="px-3 pb-3 text-sm text-gray-400 leading-relaxed border-t border-white/5 pt-2">
                    {item.a}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Contact / Support */}
        <div className="bg-panel rounded-xl p-5 border border-white/5">
          <SectionHeader icon="💬" title="Need More Help?" />
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              { title: "Documentation", desc: "Read the full technical documentation and API reference.", icon: "📖", link: "#" },
              { title: "GitHub", desc: "View the source code, report issues, or contribute.", icon: "🐙", link: "#" },
              { title: "Community", desc: "Join the community for discussions and support.", icon: "👥", link: "#" },
            ].map((item) => (
              <a key={item.title} href={item.link}
                className="bg-[#0a0c12] rounded-xl p-4 border border-white/5 hover:border-accent/30 transition group block">
                <span className="text-2xl">{item.icon}</span>
                <h4 className="text-white font-semibold text-sm mt-2 group-hover:text-accent transition">{item.title}</h4>
                <p className="text-gray-500 text-xs mt-1">{item.desc}</p>
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
