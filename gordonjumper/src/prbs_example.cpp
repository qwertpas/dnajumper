// #include <Arduino.h>
// #include <math.h>
// #include <stdint.h>

// // ================== USER PLACEHOLDERS ==================
// // Command motor driver voltage in [-7, +7] volts.
// // Implement with your DAC/PWM->driver interface.
// static void set_voltage(float volts) {
//   // TODO: implement
//   (void)volts;
// }

// // Read speed in rad/s (already converted).
// // Implement using encoder, Hall, etc.
// static float read_speed_rad_s() {
//   // TODO: implement
//   return 0.0f;
// }
// // =======================================================

// // Tiny PRNG: xorshift32 (deterministic, fast)
// static uint32_t xorshift32(uint32_t &state) {
//   state ^= state << 13;
//   state ^= state >> 17;
//   state ^= state << 5;
//   return state;
// }

// void setup() {
//   Serial.begin(2000000);  // high baud helps 1 kHz logging
//   delay(200);

//   // ---- Experiment params ----
//   const float V0 = 5.0f;               // amplitude (V), keep within ±7
//   const float V_LIMIT = 7.0f;          // clamp
//   const uint32_t BIT_US = 20000;       // 20 ms PRBS bit time
//   const uint32_t SAMPLE_US = 1000;     // 1 kHz logging
//   const uint32_t DURATION_US = 10'000'000; // 10 s run

//   // Optional: short settle time at 0V
//   set_voltage(0.0f);
//   delay(200);

//   uint32_t rng = 0x12345678u;
//   if (rng == 0) rng = 1;

//   uint32_t t0 = micros();
//   uint32_t next_bit = t0;
//   uint32_t next_sample = t0;

//   float v_cmd = 0.0f;

//   // CSV header
//   Serial.println("t_us,v_cmd,omega_rad_s");

//   while ((uint32_t)(micros() - t0) < DURATION_US) {
//     uint32_t now = micros();

//     // Update PRBS each bit
//     if ((int32_t)(now - next_bit) >= 0) {
//       next_bit += BIT_US;

//       uint32_t r = xorshift32(rng);
//       v_cmd = (r & 1u) ? +V0 : -V0;

//       // Clamp
//       if (v_cmd >  V_LIMIT) v_cmd =  V_LIMIT;
//       if (v_cmd < -V_LIMIT) v_cmd = -V_LIMIT;

//       set_voltage(v_cmd);
//     }

//     // Sample / log at ~1 kHz
//     if ((int32_t)(now - next_sample) >= 0) {
//       next_sample += SAMPLE_US;

//       float omega = read_speed_rad_s();

//       // CSV row: time since start (us), commanded voltage, speed (rad/s)
//       // Using printf is convenient; if it's slow on your core, see note below.
//       Serial.printf("%u,%.3f,%.6f\n", (unsigned)(now - t0), v_cmd, omega);
//     }

//     // Optional: yield to WiFi/RTOS; keep tiny so timing doesn't drift much
//     // delayMicroseconds(50);
//   }

//   // Stop motor
//   set_voltage(0.0f);
//   Serial.println("# done");
// }

// void loop() {
//   // nothing
// }
