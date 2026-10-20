# W6 same0419 native single-trial matrix

|Arm|NativeC2(s)|ExcludedC1(s)|Peakallocated(GiB)|Peakreserved(GiB)|Samplednodecgrouppeak(GiB)|
|---|---:|---:|---:|---:|---:|
|resident_named_CPU_fallback|308.670470170|315.127993127|74.158217|76.730469|859.847759|
|fsdp|320.666055212|322.936444398|42.881851|46.566406|749.584553|
|candidate|278.886441949|359.380386299|41.140718|46.140625|1256.214840|

Source 0419b96b9ed9a8dd76737042e6c5020c80f6673a. FSDP→candidate descriptive ratio1.149808693x; FA2<=280s target reached,1.5x not reached.
All8physical/16independent logicalslots,2cyclesfinalStudent2/Fake2/EMA0,census/runtime/versions/leasePASS,zero newOOM andowncleanupempty.
Resident CPUparameter+boundary staging is namedcapacityfallbackLOW_HEADROOM (~0.458GiBminimumsampledfree), NOTpureGPUResident.
Sampledcgroupmemoryincludesfilecache/kernel/anon, not summedmodelRSS; kernelmemory.peakAPIunavailable.
Oldactual27008strictbitwiseC1at1abd plusdefault-offentireworkerASTproof preserved. No fresh0419fullmodelcrossarmstateparity andnoC2fullstateproof. No formal numerical equivalence/speedup claim orCVfromsingletrial.
No Nsight in these measurements; profilingtimes excluded. No A100or external190sMFU claim.
