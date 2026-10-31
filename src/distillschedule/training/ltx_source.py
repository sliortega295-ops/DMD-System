"""Opt-in hash-pinned ACTUAL LTX-2.5 source, never a substitute.

LTX-2.x is externally licensed under its community license. No upstream source
or weights are bundled or relicensed. Source-only discovery is Torch-free;
original() imports only the caller-selected local source closure.
"""
from __future__ import annotations

from types import SimpleNamespace

from .upstream import PinnedSource

REPOSITORY = "https://github.com/Lightricks/LTX-2"
REVISION = "e98f6d927d83c92ef2be6aec76c8440b19306e39"
SOURCE_MANIFEST_SHA256 = "6e3539b8134e81a3aeb5815f46c8a6a8af147120f1b9d25eb0050635de92cf7b"
LICENSE = "LTX-2.x Community License (external LICENSE-2_x; not Apache-2.0)"
SOURCE_FILES = {
    "LICENSE-2_x": "ee5b0c880c1516e15bd51c8576a05ea1d2c6e5bb2d070ca098e847e42a75af34",
    "packages/ltx-core/pyproject.toml": "efc37192d798273912e459e04ea4deeae6923e349869ec04ac3f1acf1082e9cb",
    "packages/ltx-core/src/ltx_core/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "packages/ltx-core/src/ltx_core/color/yuv.py": "8c80b30032a1dcc70e36ddda2b6bf5da937ad86f79e861befec84bc1fc7f2479",
    "packages/ltx-core/src/ltx_core/components/__init__.py": "a48f5006c52ea1e08f8c463461354d4ddedd50d331c183d5099f5b3bd61f1425",
    "packages/ltx-core/src/ltx_core/components/patchifiers.py": "9618d6adac114b8b21a91c53a582ae19ab944813b25c88ffc11a1aa7e0239e0a",
    "packages/ltx-core/src/ltx_core/components/protocols.py": "d1d79c9507c57b786ff752aedc8fc988b24895ef81eb95f31506cdcb8253dbc3",
    "packages/ltx-core/src/ltx_core/components/schedulers.py": "ca318b9e9d0e3166e7085b8abf1950a2851790f9427bfdcb8e825cc6a8d06be4",
    "packages/ltx-core/src/ltx_core/devices.py": "4e3992e9fa77d1bc8f0ce9d58f782aa71bf3d4e0d72da3d7e12d51faaf4eae83",
    "packages/ltx-core/src/ltx_core/guidance/__init__.py": "87d78713485160abab95edb4eae52a4a65d037e1f4ab26369fa228ef4e8b294e",
    "packages/ltx-core/src/ltx_core/guidance/perturbations.py": "c3a9cd73fb5f32053f9134e467306106308bf12553d4048d0dc53314fa4e3d9c",
    "packages/ltx-core/src/ltx_core/loader/__init__.py": "36ad5474ecd93700d148a5377c7eea1eec684e4ed0d265d845902da3cdfbbdbc",
    "packages/ltx-core/src/ltx_core/loader/fuse_loras.py": "08d8ccf6ad0346d7bc02496393275dffc356f8c3c232c069eda835da9601e982",
    "packages/ltx-core/src/ltx_core/loader/helpers.py": "4c274924136b46d1792bbe5e9ce297c02fc615a759814e0dd4791fac8018d214",
    "packages/ltx-core/src/ltx_core/loader/module_ops.py": "70f4a23a285fd174ddeec90580d0f73322f51ea72c54a40a57a1c99911196db8",
    "packages/ltx-core/src/ltx_core/loader/primitives.py": "ccd8e80a368ce87bd0ed8314973e52bc4176cbc1389129b300b23d8747db78f5",
    "packages/ltx-core/src/ltx_core/loader/registry.py": "7b2f3fd98245b47ae67ab6777682b5dc937587b18fda56cf45e8bd30525734ad",
    "packages/ltx-core/src/ltx_core/loader/sd_ops.py": "f3eb4639dfca1399691363370be6763eb27727907d842c21ddaddd115cda9b1e",
    "packages/ltx-core/src/ltx_core/loader/sft_loader.py": "d1578a0d2b634f303b59f61d046664d95c1ee2ba5c1a5c27b2aae759a0bfe280",
    "packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py": "85711dbef0493f2edfcbd75e9890ab72e316424052715a9747ea0a64287079d3",
    "packages/ltx-core/src/ltx_core/model/__init__.py": "012474a1c36d4fcb6a878b82fa364e7589eb9826a49cdc3a15e3cea50a9f4468",
    "packages/ltx-core/src/ltx_core/model/audio_vae/__init__.py": "3741027ec23edb22148eb939b3f2343bacb3b153d807e153f344d1f6e5a2ea44",
    "packages/ltx-core/src/ltx_core/model/audio_vae/attention.py": "8e6d04197b2d92f409e555261eca7be1c6fe177d78c80d3c806d44a2d41d7918",
    "packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py": "2e25cd5343451dd61dd189e62417987f7a058157e63c0bbae8c86947d1b0be8a",
    "packages/ltx-core/src/ltx_core/model/audio_vae/causal_conv_2d.py": "739a4793d9418df48b440b2820decbc95ad4a5d8aa878cc8a65a8fc5d1c4317c",
    "packages/ltx-core/src/ltx_core/model/audio_vae/causality_axis.py": "3d37e5bc06ed13c689c822efd25bc8a4743ea2919281cbb0780623a4b3c752a9",
    "packages/ltx-core/src/ltx_core/model/audio_vae/downsample.py": "9835e1b0b55463a2b9edc49fb73e628ab303d37cb714c32bc6db10436f8531aa",
    "packages/ltx-core/src/ltx_core/model/audio_vae/model_configurator.py": "351d4b2bc0aa507ed5118262c4293c88d92a03c87aceb0fad037e56a7704aed1",
    "packages/ltx-core/src/ltx_core/model/audio_vae/ops.py": "a3830dacb366dc6414b42754b4b450b71117db65b140b130d8e3eb69b8668321",
    "packages/ltx-core/src/ltx_core/model/audio_vae/resnet.py": "00e571dd7f78ba455d94686b8cf1c8610cf24da09167a62fb2e2c4e3d8117c0e",
    "packages/ltx-core/src/ltx_core/model/audio_vae/upsample.py": "40dea5e410aa17eb1c070d42a2b13d3da010beef5823170f5549af4068607c5f",
    "packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py": "443d34c61ad5204cefb6adf5d481a02dbb5c7b9115f2ce6f731296044c049489",
    "packages/ltx-core/src/ltx_core/model/common/normalization.py": "e0cc7179f1e00d8a517c8c8a61ee0f2f8dd6476fe21c69175b9fd57bc35e840b",
    "packages/ltx-core/src/ltx_core/model/disposable.py": "4aa42f942ed3481e82edd42816e338f9d7eac78479ebc84fcccf0d03c9a29916",
    "packages/ltx-core/src/ltx_core/model/model_protocol.py": "982363d6bb663685ad66e97be339b6daeeeda757380fa21014e1cf93d376be99",
    "packages/ltx-core/src/ltx_core/model/transformer/__init__.py": "df393b94108b18e45cbf3d779ff4eed6d8d7ba854b9cc2304a82a15874dff52d",
    "packages/ltx-core/src/ltx_core/model/transformer/adaln.py": "87be8b1ec9b1395156d2e0f22834ed993af31d684198109d33e61c68f53cd065",
    "packages/ltx-core/src/ltx_core/model/transformer/attention.py": "e65abeb1536d0ae8b0f1236a913f2b5675fab61ba09d90fe33ccc03f496cabbd",
    "packages/ltx-core/src/ltx_core/model/transformer/compiling.py": "9b50d6af0a2e4d0c577a8913e499593ad47cc3a50ad15df23063ed0bb99a34d2",
    "packages/ltx-core/src/ltx_core/model/transformer/cudagraph_capture.py": "ed80f356b43f5e8cc87cead412f2d4ea5c724640d9b7d16f841208224f43a8bd",
    "packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py": "e6c743950e90b0540499674c688271c5aab45101e6f776fbdde78f50bff12567",
    "packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py": "c9d008a2db4f945ee39d24ff6c191fc6c15093ad021d86648b6a70dc16d6da55",
    "packages/ltx-core/src/ltx_core/model/transformer/modality.py": "b37d99f659b4e4888ba9cdaed562890648105e6b158245e97ba3ee9123be22db",
    "packages/ltx-core/src/ltx_core/model/transformer/model.py": "49b841c389c70872b5858b3d7fcca3c97a78bca8c5646976523217366b0efbaa",
    "packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py": "8e9c6c0838ceed2eeac480eb1b13c8f22bfbbdf068b34df720888ab28926c625",
    "packages/ltx-core/src/ltx_core/model/transformer/rope.py": "6be5f15cc736544a49aa775fdf0befea35be3f10a9a2977160c0eaeb5bbd9fe4",
    "packages/ltx-core/src/ltx_core/model/transformer/text_projection.py": "989cd23bfd6491328625a5850f8212afda855418570f1059b20377415b8b2e7a",
    "packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py": "3a75ff5ef6231bed408f10279a4fb3f0b76c7b027ebaf0aae8c0255fb5338b97",
    "packages/ltx-core/src/ltx_core/model/transformer/transformer.py": "6d24b7294be9afb28d69156b9a17d59477d01b43547156cf79767cee1f9f24d5",
    "packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py": "dbb97d200f6de3f82155638978294d30227adaa9131736b6970ff270509b72da",
    "packages/ltx-core/src/ltx_core/model/video_vae/__init__.py": "dd2c8804ccafbc97c7c6b28b6c7269bdc39c3cadb8b14c4f7b38a90c211ecac9",
    "packages/ltx-core/src/ltx_core/model/video_vae/attention.py": "c6f5b5a60a429313068d9d2622b54e01393d92cf02b23fa12fee019aaaad1595",
    "packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py": "cd57d82637bdbb1d737f6f9c510525040ea01141eb765d4a7530422139fb5f82",
    "packages/ltx-core/src/ltx_core/model/video_vae/convolution.py": "38e6d37eff8c8feb4d5d363d2764173c8e4534de5d5ec19411c387382754e0f8",
    "packages/ltx-core/src/ltx_core/model/video_vae/decode_workspace.py": "b6a45021fd1b184dc24c93a1108229b2d648f5c4eae09994f5b9cc4aca65bc37",
    "packages/ltx-core/src/ltx_core/model/video_vae/diffusion_tiling.py": "8dc8a67440873a7bd635ab6c15c9d6ac14296bead23f16baa385acb416489bdb",
    "packages/ltx-core/src/ltx_core/model/video_vae/diffusion_video_decoder.py": "428a1ae4512622e17e1b91d47bf8695b1c4f6aa94a7b4386aafb77fc8b5ba6da",
    "packages/ltx-core/src/ltx_core/model/video_vae/enums.py": "9e9f9fc83f0f76ab1022576a52c65b5ba8dc99f5ad872e7de372333479d3bedd",
    "packages/ltx-core/src/ltx_core/model/video_vae/keyframes.py": "ecc5416fe9fe28e57b87d4436a96919e9b5de8f634d796caf49e421329d9a664",
    "packages/ltx-core/src/ltx_core/model/video_vae/memory_efficient_decode.py": "4fd5a47c3eab20af77000d2d7293808d7feefc7162c0e52a285d5775f12f85d4",
    "packages/ltx-core/src/ltx_core/model/video_vae/model_configurator.py": "ef07c503237aba86647b3c09fe9cef8921b3e2fbfabbcf9e2af2a0892817d243",
    "packages/ltx-core/src/ltx_core/model/video_vae/ops.py": "d0796c448e33b060f633e60243839fe74e8266b61e4aee618a1974b15adc580e",
    "packages/ltx-core/src/ltx_core/model/video_vae/resnet.py": "347924a483610c3fb66da2dc8dd2d9ed323cb6b8fe8d43b44f587323e1a7b852",
    "packages/ltx-core/src/ltx_core/model/video_vae/sampling.py": "226bec4578ba2b0895267c10ad4845762e2cdc39d96c582f7fac5eb7a5adeada",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/__init__.py": "52042416ae3197363e60765d08cbca6e5f142c6e7c40985cd7e68a5af39d67a4",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/apply.py": "8074297bb11e29a5c74c7df797c95472e715134dbce985bea6c973d4ea139a5b",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/attention.py": "9eba8e3ddda157b813b0986da0d57748abb56cbf132b0ffdaabdea1375f6d98c",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/blocks.py": "b1a1bac038fac7c85a48fb277535b263e949cf06cd4385822af7274bf2903c6a",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/chunked/attn.py": "d5ea966a17b81576d8d5be188c62a4f919e7db619d624cacad44e4f5f7edd9e1",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/chunked/block.py": "3543a88dc152556c1f39d284f9827bca1b4a4b8523ee939ca56352c9469b2539",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/chunked/context.py": "da69205d3ee2330cd8568123f4ac10bb257688974541e50ea20e8473601ec0b7",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/chunked/mlp.py": "e707f933b7b757f005587b170513b6fe88e1f876e170f773d21167ea3b034839",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/combined/attn.py": "350ef23e7fd81ba1d50e6be56c60a1f7bb4ef1413f052baa47d35729afd68136",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/combined/block.py": "0068d9fe790620d10a65cc6b7d952b8d3c0e5ec4751437c79e8d268dfccb2d3d",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/combined/context.py": "89cd0f273f3e30770a8a56aa7b200df4560ac645a58f0778ac073bfcf1cb7481",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/combined/mlp.py": "c73e25f52aaa91fd41b3b03217134e6383d0aef32dc9a4750723581da30adbae",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/compiling.py": "f0fe44831c85d8bd0c5536423574573f3a3c70e410a5ff588b43b20ca81972d6",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/config.py": "07b6301a0e7799553c27dcb2e7b6f33bc829febc983c5f0092173539fb2b13ae",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/det_attn_rope.py": "ee61729e93c266ebef28b07ad78f8203748a4a5895ec1ee2377e0577d8a90fdb",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/det_rms_norm.py": "5b05ac0eead977aaa9e7bd4a827e0411f3d318656ba183743848b4d6e0ef9417",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/dsl_kernels/__init__.py": "29e526da9858fdd1158a360ab6dbb62085f5aee9b36c5a797a59e41717e1f887",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/dsl_kernels/apply_dsl.py": "d117e9dd3bdc73c2e91342c2e90a3885b4d1438a57c62bfaf0a034b4d317f0a0",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/dsl_kernels/attn.py": "e2530714f92937936e786ebb7bb01520645ac2c3751d1b719b488142c425ba0b",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/dsl_kernels/block.py": "3b099b822ba2406344e10b4ab6fe313c163fa3cfd61620d939112f03226527ae",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/dsl_kernels/chain.py": "4d179a39aabe01ec0f83dd90d67fe2ed3103bd415c9caac79ebbe6bce43ac1cf",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/dsl_kernels/joint_attn.py": "9994e0b463bff3f33670fcf64186c0652820f59a81b353ea8fd7f0b093422fa7",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/__init__.py": "ed5f3eb4cc7c43613a890d7b678951ba896fb02537a84c6574c7dc651af57d34",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/eager.py": "287c6eaadeded70150d390116bac845e6798579abd94a7acb0dc65e46fb1983c",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/joint_eager.py": "5f830fa78022d727db1205bfae59f9b6c2bfcf759d349922222866476cd58fce",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/joint_triton.py": "3ec51d72c5bbd0bafa7bde8494f369c26a5c6e5b9a6598468d6768b12a9fe705",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/fallback_na/triton_na.py": "02884a9a105b7397998e17a63723a7bf2558f3ceae55e51aa5373df212e90af7",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/layers.py": "184c73dff1e68536cbb826f082b5f572344ca7cae15ab80a73edb051ec14710d",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/qkv.py": "e8206a507b33e8ccd401d585eeaecda43f9abddbefca52418d459901b3c4ca50",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/rope.py": "c9ff9541799b7ced2496c249514d9aa2c4d59a436769c83ec5d658ba166ce028",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/rope_math.py": "3e65d86aa04c5ac70359018680318eda1ccbe52a1f4e130930fa3046b2297557",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/row_rms_norm.py": "a71fea9164cdb6fd3a3ab9f3dd4cc31a5168d967e6b6a52283a484797badc5fe",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/stage5_tail.py": "50f81936b682d2015f8f4b073369db45020a5d8aa63b0d3d3120623b5e20ff68",
    "packages/ltx-core/src/ltx_core/model/video_vae/transformer/swiglu.py": "6e404bb8ed42c8d847fe0dc58ae39bcc081cfdc53b7eccfe49a078f079bed1a7",
    "packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py": "81e305e92482e03b4809311a96a619933b47c1e2d53c3dc34373c5df30a71ee7",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/__init__.py": "412310a34f014043aa58d215b0c4b802625fd89e55ab066b8b184bd854a27182",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py": "820564f138f827686304387431d106f37d75f9dfc66de4e1b8034c9a32871454",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py": "b724ff0f7bfe57e6d82f02b823be2668ca48e7a3f0f3d71337679a6e387e317e",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py": "691286d79e42749513efdac67fcaac26c549c704bec871759fb5c38a2473850a",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py": "4bcf35d714c7c481450c30264fa94b65c6bd8fb2897932cd10009cd9f0b3e433",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py": "76c1a5975b31f3d2a47a29149ef7978024f213b6da0b09f2d94564895d3834d6",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/gemma_assets.py": "b9ffd8657610d3b37882bbe366edc455eb0ddcd2254577e7a9a7eb14926eb704",
    "packages/ltx-core/src/ltx_core/text_encoders/gemma/tokenizer.py": "4f794626d9ce7113f1652c682fa1ea1228544a719c0ebf34d13a66501ed59e79",
    "packages/ltx-core/src/ltx_core/tiling.py": "1ab828d2e9f316f8ecce53b41406046c0da35cb4d446734aeaceea440c13b0cf",
    "packages/ltx-core/src/ltx_core/types.py": "b038620f6c9262325139165befcc2e33d497b8157a6f3031d228802b8b6db2f5",
    "packages/ltx-core/src/ltx_core/utils.py": "d4acb090aa380165e47739ec5c9dc89d58e9ab2b20e7db06c45c1c7554ab4810",
    "packages/ltx-trainer/src/ltx_trainer/model_loader.py": "6a6fdf5847521308869f9fda8184764495a33df053055f8a9fbcbd7308ef2ff6",
    "packages/ltx-trainer/src/ltx_trainer/trainer.py": "7bb4441f785b20350656056206a48132bbfc676cce5d58b0665c91b295f3ac01",
    "packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py": "265bcc6e018224156218844a94c54155cba734f8fc038d5eeb09f28e619761bb",
    "packages/ltx-trainer/src/ltx_trainer/training_strategies/text_to_video.py": "d675898794d7f0f90aea988f7bf913f0bc62874b5be2dc0a0d92259bf1c0c375"
}


def require(condition, message):
    if not condition:
        raise ValueError("LTX-2.5: " + message)


def original(source_root):
    source = PinnedSource(source_root, SOURCE_FILES, import_roots=("ltx_core",),
                          python_roots=("packages/ltx-core/src",))
    attention = source.import_module("ltx_core.model.transformer.attention")
    # Native text connectors and the original checkpoint configurator do not
    # expose AttentionOps. Their AUTOMATIC constructor otherwise probes CUDA0
    # even while creating CPU/meta modules. This private, hash-locked namespace
    # is our explicit reference integration, not the user's normal ltx_core.
    # Reuse the EXACT native CPU/Ampere SDPA priority factory; do not hide CUDA,
    # patch Torch, rewrite an attention forward, or change its backend order.
    # The actual Transformer has a separate explicit MATH pin at its boundary.
    attention.automatic_attention = attention._sdpa_full_priority
    attention.automatic_masked_attention = attention._sdpa_full_priority
    return SimpleNamespace(source=source,
        identity=dict(repository=REPOSITORY, revision=REVISION, license=LICENSE,
            default_attention="NATIVE_SDPA_FULL_PRIORITY_WITHOUT_DEVICE_PROBE", **source.identity),
        model=source.import_module("ltx_core.model.transformer.model"),
        attention=attention,
        modality=source.import_module("ltx_core.model.transformer.modality"),
        patchifiers=source.import_module("ltx_core.components.patchifiers"),
        types=source.import_module("ltx_core.types"))
