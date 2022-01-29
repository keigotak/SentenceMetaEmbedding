from TrainVectorAttentionWithSTSBenchmark import TrainVectorAttentionWithSTSBenchmark


trainer = TrainVectorAttentionWithSTSBenchmark(device='cpu')
tag = '12052021102319620233' # '11222021182523445587' # '10302021131616868619' # '10272021232254714917' # '10252021190301856515' # 10222021201617472745, , 10192021082737054376
trainer.set_tag(tag)
trainer.load_model()
print(tag)
