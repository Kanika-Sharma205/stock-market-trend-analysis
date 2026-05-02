import Header from './components/Header'
import HeroSection from './components/HeroSection'
import ProblemStatement from './components/ProblemStatement'
import Approach from './components/Approach'
import ModelPerformance from './components/ModelPerformance'
import CustomPrediction from './components/CustomPrediction'
import Footer from './components/Footer'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950">
      <Header />
      <main>
        <HeroSection />
        <ProblemStatement />
        <Approach />
        <ModelPerformance />
        <CustomPrediction />
      </main>
      <Footer />
    </div>
  )
}
