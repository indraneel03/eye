import React from 'react';
import { motion } from 'framer-motion';
import { Mail, Phone } from 'lucide-react';

const AboutPage = () => {
  return (
    <div
      className="relative bg-cover bg-center min-h-screen"
      style={{
        backgroundImage:
          "url('https://static.vecteezy.com/system/resources/previews/053/793/126/non_2x/close-up-view-of-a-horses-eye-showcasing-its-unique-features-and-natural-beauty-free-photo.jpeg')",
      }}
    >
      {/* Dark overlay for better text visibility */}
      <div className="absolute inset-0 bg-gradient-to-r from-black/70 to-black/50"></div>

      <div className="relative max-w-6xl mx-auto p-6 text-white">
        <motion.h1
          className="text-5xl font-extrabold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          About OCT Analysis
        </motion.h1>

        <motion.div
          className="bg-white/10 backdrop-blur-sm rounded-lg p-8 shadow-2xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <p className="text-lg leading-relaxed mb-6">
            OCT Analysis is an advanced retinal disease detection platform powered by AI. Our mission is to provide accurate and efficient diagnosis of retinal diseases using state-of-the-art deep learning models.
          </p>

          <h2 className="text-2xl font-semibold mb-4">Our Technology</h2>
          <p className="text-lg leading-relaxed mb-6">
            We leverage the latest advancements in deep learning and computer vision to analyze Optical Coherence Tomography (OCT) images. Our models are trained on large datasets of OCT images to detect various retinal conditions with high accuracy.
          </p>

          <h2 className="text-2xl font-semibold mb-4">Understanding Retinal Diseases</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {/* CNV Section */}
            <motion.div
              className="relative bg-cover bg-center rounded-lg p-6 shadow-lg overflow-hidden"
              style={{
                backgroundImage: "url('https://www.aao.org/image.axd?id=5b5b5b5b-5b5b-5b5b-5b5b-5b5b5b5b5b5b&t=637685000000000000')",
              }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <div className="absolute inset-0 bg-black/50"></div>
              <div className="relative z-10">
                <h3 className="text-xl font-semibold mb-3">Choroidal Neovascularization (CNV)</h3>
                <p className="text-lg leading-relaxed mb-3">
                  CNV is a condition where new blood vessels grow from the choroid layer of the eye into the retina, often leading to vision loss. It is commonly associated with age-related macular degeneration (AMD).
                </p>
                <p className="text-lg leading-relaxed mb-3">
                  <strong>How it occurs:</strong> CNV occurs due to the breakdown of the retinal pigment epithelium (RPE), which allows abnormal blood vessels to grow.
                </p>
                <p className="text-lg leading-relaxed">
                  <strong>Mitigation:</strong> Treatment options include anti-VEGF injections, laser therapy, and photodynamic therapy to reduce abnormal blood vessel growth.
                </p>
              </div>
            </motion.div>

            {/* DME Section */}
            <motion.div
              className="relative bg-cover bg-center rounded-lg p-6 shadow-lg overflow-hidden"
              style={{
                backgroundImage: "url('https://www.aao.org/image.axd?id=5b5b5b5b-5b5b-5b5b-5b5b-5b5b5b5b5b5b&t=637685000000000000')",
              }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              <div className="absolute inset-0 bg-black/50"></div>
              <div className="relative z-10">
                <h3 className="text-xl font-semibold mb-3">Diabetic Macular Edema (DME)</h3>
                <p className="text-lg leading-relaxed mb-3">
                  DME is a complication of diabetic retinopathy where fluid accumulates in the macula, leading to swelling and vision impairment.
                </p>
                <p className="text-lg leading-relaxed mb-3">
                  <strong>How it occurs:</strong> DME occurs due to the leakage of blood vessels in the retina caused by high blood sugar levels in diabetic patients.
                </p>
                <p className="text-lg leading-relaxed">
                  <strong>Mitigation:</strong> Management includes anti-VEGF injections, corticosteroids, and laser treatment to reduce swelling and improve vision.
                </p>
              </div>
            </motion.div>

            {/* Drusen Section */}
            <motion.div
              className="relative bg-cover bg-center rounded-lg p-6 shadow-lg overflow-hidden"
              style={{
                backgroundImage: "url('https://www.aao.org/image.axd?id=5b5b5b5b-5b5b-5b5b-5b5b-5b5b5b5b5b5b&t=637685000000000000')",
              }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.8 }}
            >
              <div className="absolute inset-0 bg-black/50"></div>
              <div className="relative z-10">
                <h3 className="text-xl font-semibold mb-3">Drusen</h3>
                <p className="text-lg leading-relaxed mb-3">
                  Drusen are tiny yellow or white deposits that form in the retina, often associated with aging and AMD.
                </p>
                <p className="text-lg leading-relaxed mb-3">
                  <strong>How it occurs:</strong> Drusen are formed by the accumulation of waste products from retinal cells, which are not properly cleared away.
                </p>
                <p className="text-lg leading-relaxed">
                  <strong>Mitigation:</strong> Regular eye exams and lifestyle changes such as a healthy diet and avoiding smoking can help manage drusen. In some cases, supplements like AREDS2 may be recommended.
                </p>
              </div>
            </motion.div>

            {/* Normal Retina Section */}
            <motion.div
              className="relative bg-cover bg-center rounded-lg p-6 shadow-lg overflow-hidden"
              style={{
                backgroundImage: "url('https://www.aao.org/image.axd?id=5b5b5b5b-5b5b-5b5b-5b5b-5b5b5b5b5b5b&t=637685000000000000')",
              }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 1.0 }}
            >
              <div className="absolute inset-0 bg-black/50"></div>
              <div className="relative z-10">
                <h3 className="text-xl font-semibold mb-3">Normal Retina</h3>
                <p className="text-lg leading-relaxed mb-3">
                  A normal retina is healthy and free from any significant abnormalities or diseases.
                </p>
                <p className="text-lg leading-relaxed mb-3">
                  <strong>How it occurs:</strong> A normal retina is maintained through good eye health practices, regular check-ups, and a healthy lifestyle.
                </p>
                <p className="text-lg leading-relaxed">
                  <strong>Mitigation:</strong> To maintain a normal retina, it is important to have regular eye exams, protect your eyes from UV light, and maintain a balanced diet rich in vitamins and minerals.
                </p>
              </div>
            </motion.div>
          </div>

          <h2 className="text-2xl font-semibold mb-4">Contact Us</h2>
          <p className="text-lg leading-relaxed mb-4">
            If you have any questions or would like to learn more about our platform, please reach out to us:
          </p>
          <ul className="list-disc list-inside text-lg">
            <li className="mb-2">
              <a
                href="mailto:indraneelkalva@gmail.com"
                className="flex items-center text-blue-400 hover:text-blue-300 transition-colors duration-300"
              >
                <Mail className="mr-2" />
                <span className="font-semibold">Email:</span> indraneelkalva@gmail.com
              </a>
            </li>
            <li>
              <a
                href="tel:7730841030"
                className="flex items-center text-blue-400 hover:text-blue-300 transition-colors duration-300"
              >
                <Phone className="mr-2" />
                <span className="font-semibold">Phone:</span> 7730841030
              </a>
            </li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
};

export default AboutPage;