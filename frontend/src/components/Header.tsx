export default function Header() {
  const links = [
    { href: '#problem',     label: 'Problem' },
    { href: '#approach',    label: 'Approach' },
    { href: '#performance', label: 'Performance' },
    { href: '#predict',     label: 'Predict' },
  ]

  return (
    <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur border-b border-slate-800/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between h-16">
        <a href="#" className="flex items-center gap-3">
          <img src="/logo.svg" alt="Logo" className="h-9 w-auto" />
        </a>
        <nav className="hidden md:flex items-center gap-1">
          {links.map(l => (
            <a
              key={l.href}
              href={l.href}
              className="px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              {l.label}
            </a>
          ))}
        </nav>
        <a
          href="#predict"
          className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-white text-sm font-semibold transition-colors"
        >
          Try Prediction
        </a>
      </div>
    </header>
  )
}
