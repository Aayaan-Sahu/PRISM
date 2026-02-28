import { useEffect, useState } from 'react'

const BARS = Array.from({ length: 32 }, () => ({
  minh: `${4  + Math.random() * 14}px`,
  maxh: `${22 + Math.random() * 36}px`,
  dur:  `${0.25 + Math.random() * 0.45}s`,
  del:  `${-(Math.random() * 2)}s`,
}))

export default function App() {
  const [locked,   setLocked]   = useState(false)
  const [keyDown,  setKeyDown]  = useState(false)

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault()
        setLocked(true)
        setKeyDown(true)
      }
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setLocked(false)
        setKeyDown(false)
      }
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup',   up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup',   up)
    }
  }, [])

  return (
    <div className="app">

      <div className={`status${locked ? ' locked' : ''}`}>
        {locked ? 'voice locked' : 'listening'}
      </div>

      <div className="indicator">
        <div className={`ring${locked ? ' active' : ''}`} />
        {!locked && <div className="ring-pulse" />}
        <div className={`dot${locked ? ' active' : ''}`} />
      </div>

      <div className="waveform">
        {BARS.map((b, i) => (
          <div
            key={i}
            className={`bar${locked ? ' locked' : ''}`}
            style={{
              '--minh': b.minh,
              '--maxh': b.maxh,
              animationDuration: b.dur,
              animationDelay:    b.del,
            } as React.CSSProperties}
          />
        ))}
      </div>

      <div className={`hint${locked ? ' locked' : ''}`}>
        <span className={`key${keyDown ? ' pressed' : ''}`}>space</span>
        {locked ? 'release to resume' : 'hold to lock voice'}
      </div>

    </div>
  )
}
