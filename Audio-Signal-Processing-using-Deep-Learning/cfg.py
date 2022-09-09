class Config:
	def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
		self.mode = mode
		self.nfilt = nfilt
		self.nfeat = nfeat
		self.nfft = nfft
		self.rate = rate
		self.step = int(rate/10)