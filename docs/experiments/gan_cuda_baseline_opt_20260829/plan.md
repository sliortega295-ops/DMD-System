# Executable plan

1. Finish overlap-aware phase attribution for both node reports and preserve
   the remote JSON evidence.
2. Import the exact compute-fusion substrate through `6005c03` without taking
   its node-HSDP or No-GAN defaults.
3. Run focused tests and record the exact candidate diff/receipts.
4. Run one frozen GAN-on world16 correctness gate.
5. If correctness passes, run one unprofiled timing gate and promote/reject C1.
6. If C1 remains above 121.110138103 s, profile only the remaining hotspot and
   iterate on the next single evidence-backed candidate.

