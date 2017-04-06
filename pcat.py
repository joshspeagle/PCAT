import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_float
import matplotlib.pyplot as plt
import time

def psf_poly_fit(psf0, nbin):
	assert psf0.shape[0] == psf0.shape[1] # assert PSF is square
	npix = psf0.shape[0]

	# pad by one row and one column
	psf = np.zeros((npix+1, npix+1), dtype=np.float32)
	psf[0:npix, 0:npix] = psf0
	
	# make design matrix for each nbin x nbin region
	nc = npix/nbin # dimension of original psf
	nx = nbin+1
	y, x = np.mgrid[0:nx, 0:nx] / np.float32(nbin)
	x = x.flatten()
	y = y.flatten()
	A = np.array([np.full(nx*nx, 1, dtype=np.float32), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y], dtype=np.float32).T
	# output array of coefficients
	cf = np.zeros((nc, nc, A.shape[1]), dtype=np.float32)

	# loop over original psf pixels and get fit coefficients
	for iy in xrange(nc):
	 for ix in xrange(nc):
		# solve p = A cf for cf
		p = psf[iy*nbin:(iy+1)*nbin+1, ix*nbin:(ix+1)*nbin+1].flatten()
		AtAinv = np.linalg.inv(np.dot(A.T, A))
		ans = np.dot(AtAinv, np.dot(A.T, p))
		cf[iy,ix,:] = ans
	
	return cf

def image_model_eval(x, y, f, back, imsz, cf, weights=None, ref=None, lib=None):
	assert x.dtype == np.float32
	assert y.dtype == np.float32
	assert f.dtype == np.float32
	assert cf.dtype == np.float32
	if ref is not None:
		assert ref.dtype == np.float32

	if weights is None:
		weights = np.full(imsz, 1., dtype=np.float32)

	nstar = x.size
	nc = 25 # should pass this in
	rad = nc/2 # 12 for nc = 25

	ix = np.ceil(x).astype(np.int32)
	dx = ix - x
	iy = np.ceil(y).astype(np.int32)
	dy = iy - y

	dd = np.stack((np.full(nstar, 1., dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f

	if lib is None:
		image = np.full((imsz[1]+2*rad+1,imsz[0]+2*rad+1), back, dtype=np.float32)
		recon = np.dot(dd.T, cf.T).reshape((nstar,nc,nc))
		for i in xrange(nstar):
			image[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

		image = image[rad:imsz[1]+rad,rad:imsz[0]+rad]

		if ref is not None:
			diff = ref - image
			diff2 = np.sum(diff*diff*weights)
	else:
		image = np.full((imsz[1], imsz[0]), back, dtype=np.float32)
		recon = np.zeros((nstar,nc*nc), dtype=np.float32)
		reftemp = ref
		if ref is None:
			reftemp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
		diff2 = lib(imsz[0], imsz[1], nstar, nc, cf.shape[1], dd, cf, recon, ix, iy, image, reftemp, weights)

	if ref is not None:
		return image, diff2
	else:
		return image

# ix, iy = 0. to 3.999
def testpsf(cf, psf, ix, iy, lib=None):
	psf0 = image_model_eval(np.array([12.-ix/5.], dtype=np.float32), np.array([12.-iy/5.], dtype=np.float32), np.array([1.], dtype=np.float32), 0., (25,25), cf, lib=lib)
	plt.subplot(2,2,1)
	plt.imshow(psf0, interpolation='none', origin='lower')
	plt.title('matrix multiply PSF')
	plt.subplot(2,2,2)
	iix = int(np.floor(ix))
	iiy = int(np.floor(iy))
	dix = ix - iix
	diy = iy - iiy
	f00 = psf[iiy:125:5,  iix:125:5]
	f01 = psf[iiy+1:125:5,iix:125:5]
	f10 = psf[iiy:125:5,  iix+1:125:5]
	f11 = psf[iiy+1:125:5,iix+1:125:5]
	realpsf = f00*(1.-dix)*(1.-diy) + f10*dix*(1.-diy) + f01*(1.-dix)*diy + f11*dix*diy
	plt.imshow(realpsf, interpolation='none', origin='lower')
	plt.title('bilinear interpolate PSF')
	invrealpsf = np.zeros((25,25))
	mask = realpsf > 1e-3
	invrealpsf[mask] = 1./realpsf[mask]
	plt.subplot(2,2,3)
	plt.title('absolute difference')
	plt.imshow(psf0-realpsf, interpolation='none', origin='lower')
	plt.colorbar()
	plt.subplot(2,2,4)
	plt.imshow((psf0-realpsf)*invrealpsf, interpolation='none', origin='lower')
	plt.colorbar()
	plt.title('fractional difference')
	plt.show()

def numpairs(x,y,neigh,generate=False):
	neighx = np.abs(x[:,np.newaxis] - x[np.newaxis,:])
	neighy = np.abs(y[:,np.newaxis] - y[np.newaxis,:])
	adjacency = np.logical_and(neighx < neigh, neighy < neigh)
	nn = x.size
	adjacency[xrange(nn), xrange(nn)] = False
	pairs = np.sum(adjacency)
	print 'number of interacting pairs', pairs
	if generate:
		if pairs:
			idx = np.random.choice(x.size*x.size, p=adjacency.flatten()/float(pairs))
			i = idx / nn
			j = idx % nn
			if i > j:
				i, j = j, i
		else:
			i, j = -1, -1
		return pairs, i, j
	else:
		return pairs


psf = np.loadtxt('/n/fink1/sportillo/pcat-dnest/Data/sdss.0921_psf.txt', skiprows=1)
	# uncomment to use test "PSF" with no x or y symmetry
	#yy, xx = np.mgrid[0:125,0:125]
	#psf = xx*yy*yy*yy
cf = psf_poly_fit(psf, nbin=5)
npar = cf.shape[2]
cff = cf.reshape((cf.shape[0]*cf.shape[1], cf.shape[2]))

array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
libmmult = npct.load_library('pcat-lion', '.')
libmmult.pcat_model_eval.restype = c_float
libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float]

testpsf(cff, psf, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

# make a model to fit
imsz = (100, 100) # image size width, height
nstar = 500
truex = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
truey = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
truealpha = np.float32(2.0)
trueminf = np.float32(10.)
truelogf = np.random.exponential(scale=1./(truealpha-1.), size=nstar).astype(np.float32)
truef = trueminf * np.exp(truelogf)
trueback = np.float32(0.)
weight = np.full(imsz, 1., dtype=np.float32)

# make mock image
noise = np.random.normal(size=(imsz[1],imsz[0])).astype(np.float32)
mock = noise + image_model_eval(truex, truey, truef, trueback, imsz, cff, lib=libmmult.pcat_model_eval)

# uncomment this to use real data
'''f = open('/n/fink1/sportillo/pcat-dnest/Data/sdss.0921_pix.txt')
w, h, nband = [np.int32(i) for i in f.readline().split()]
imsz = (w, h)
assert nband == 1
junk1, junk2, junk3, junk4 = f.readline().split()
bias, gain, exposure = [np.float32(i) for i in f.readline().split()]
mock = np.loadtxt('/n/fink1/sportillo/pcat-dnest/Data/sdss.0921_cts.txt').astype(np.float32)
mock -= bias
weight = gain / mock # inverse variance
trueback = np.float32(179.) # from run-0923
CCcat = np.loadtxt('/n/fink1/sportillo/pcat-dnest/923_classical_sigfac')
truex = CCcat[:,0]
truey = CCcat[:,2]
truef = CCcat[:,4]
conf = CCcat[:,8]
trueminf = np.float32(250.)
mask = np.logical_and(conf > 0.9, truef > trueminf)
truex = truex[mask]
truey = truey[mask]
truef = truef[mask]
truealpha = np.float32(2.0) #placeholder'''

# number of stars to use in fit
nstar = 1000
n = np.random.randint(nstar)+1
x = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
y = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
f = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=nstar)).astype(np.float32)
x[n:] = 0.
y[n:] = 0.
f[n:] = 0.
back = trueback#np.float32(0.)

nsamp = 1000
nloop = 1000
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
fsample = np.zeros((nsamp, nstar), dtype=np.float32)
acceptance = np.zeros(nsamp, dtype=np.float32)
dt1 = np.zeros(nsamp, dtype=np.float32)
dt2 = np.zeros(nsamp, dtype=np.float32)

crad = 10
plt.ion()
#plt.figure(figsize=(10,5))
plt.figure(figsize=(15,5))
for j in xrange(nsamp):
	t0 = time.clock()
	nmov = np.zeros(nloop)

	resid = mock.copy() # residual for zero image is data
	model, diff2 = image_model_eval(x[0:n], y[0:n], f[0:n], back, imsz, cff, ref=resid, lib=libmmult.pcat_model_eval)
	logL = -0.5*diff2
	resid -= model

	for i in xrange(nloop):
		t1 = time.clock()
		moveweights = np.array([80., 15., 0., 15., 1.])
		moveweights /= np.sum(moveweights)
		rtype = np.random.choice(moveweights.size, p=moveweights)
		# defaults
		nw = 0
		dback = np.float32(0.)
		pn = n
		factor = 0. # best way to incorporate acceptance ratio factors?
		goodmove = False
		# mover
		if rtype == 0:
			cx = np.random.uniform()*(imsz[0]-1-2*crad)+crad
			cy = np.random.uniform()*(imsz[1]-1-2*crad)+crad
			mover = np.logical_and(np.abs(cx-x) < crad, np.abs(cy-y) < crad)
			mover[n:] = False
			nw = np.sum(mover).astype(np.int32)
			dlogf = np.random.normal(size=nw).astype(np.float32)*np.float32(0.02)
			f0 = np.extract(mover, f)
			pf = f0*np.exp(dlogf)
			dpos_rms = np.float32(50./np.sqrt(40.))/(np.maximum(f0, pf))
			dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
			dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
			px = np.extract(mover, x) + dx
			py = np.extract(mover, y) + dy
			factor = -truealpha*np.sum(dlogf)
			if (px > 0).all() and (px < imsz[0] - 1).all() and (py > 0).all() and (py < imsz[1] - 1).all() and (pf > trueminf).all():
				goodmove = (nw > 0)
		# hopper
		elif rtype == 1:
			mover = np.random.uniform(size=nstar) < 4./float(n+1)
			mover[n:] = False
			nw = np.sum(mover).astype(np.int32)
			px = np.random.uniform(size=nw).astype(np.float32)*(imsz[0]-1)
			py = np.random.uniform(size=nw).astype(np.float32)*(imsz[1]-1)
			pf = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=nw)).astype(np.float32)
			goodmove = (nw > 0)
		# background change
		elif rtype == 2:
			dback = np.float32(np.random.normal())
			mover = np.full(nstar, False, dtype=np.bool)
			nw = 0
			px = np.array([], dtype=np.float32)
			py = np.array([], dtype=np.float32)
			pf = np.array([], dtype=np.float32)
			goodmove = True 
		# birth and death
		elif rtype == 3:
			lifeordeath = np.random.randint(2)
			mover = np.full(nstar, False, dtype=np.bool)
			# birth
			if lifeordeath and n < nstar: # do not exceed n = nstar
				# append to end
				mover[n] = True
				px = np.random.uniform(size=1).astype(np.float32)*(imsz[0]-1)
				py = np.random.uniform(size=1).astype(np.float32)*(imsz[1]-1)
				pf = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=1)).astype(np.float32)
				pn = n+1
			# death
			elif not lifeordeath and n > 0: # need something to kill
				ikill = np.random.randint(n)
				mover[ikill] = True
				singlezero = np.array([0.], dtype=np.float32)
				if ikill != n-1: # put last source in killed source's place
					mover[n-1] = True
					px = np.array([x[n-1], 0], dtype=np.float32)
					py = np.array([y[n-1], 0], dtype=np.float32)
					pf = np.array([f[n-1], 0], dtype=np.float32)
				else: # or just kill the last source if we chose it
					px = singlezero
					py = singlezero
					pf = singlezero
				pn = n-1
			goodmove = True
			nw = 1
		# merges and splits
		else:
			mover = np.full(nstar, False, dtype=np.bool)
			splitsville = np.random.randint(2)
			kickrange = 1
			sum_f = 0
			low_n = 0
			pn = n
			# split
			if splitsville and n > 0 and n < nstar: # need something to split, but don't exceed nstar
				dx = np.random.uniform(-kickrange, kickrange)
				dy = np.random.uniform(-kickrange, kickrange)
				frac = np.random.uniform()
				isplit = np.random.randint(n)
				mover[isplit] = True
				mover[n] = True # split in place and add to end of array
				px = x[isplit] + np.array([(1-frac)*dx, -frac*dx], dtype=np.float32)
				py = y[isplit] + np.array([(1-frac)*dy, -frac*dy], dtype=np.float32)
				pf = f[isplit] * np.array([frac, 1-frac], dtype=np.float32)
				pn = n + 1
				if (px > 0).all() and (px < imsz[0] - 1).all() and (py > 0).all() and (py < imsz[1] - 1).all() and (pf > trueminf).all():
					goodmove = True
					# need to calculate factor
					sum_f = f[isplit]
					low_n = n
					xtemp = x.copy()
					ytemp = y.copy()
					np.place(xtemp, mover, px)
					np.place(ytemp, mover, py)
					pairs = numpairs(xtemp[0:pn], ytemp[0:pn], kickrange)
			# merge
			elif not splitsville and n > 1: # need two things to merge!
				pairs, isplit, jsplit = numpairs(x[0:n], y[0:n], kickrange, generate=True)
				if pairs:
					mover[isplit] = True
					mover[jsplit] = True
					sum_f = f[isplit] + f[jsplit]
					frac = f[isplit] / sum_f
					if jsplit != n-1: # merge to isplit and move last source to jsplit
						mover[n-1] = True
						px = np.array([frac*x[isplit]+(1-frac)*x[jsplit], x[n-1], 0], dtype=np.float32)
						py = np.array([frac*y[isplit]+(1-frac)*y[jsplit], y[n-1], 0], dtype=np.float32)
						pf = np.array([f[isplit] + f[jsplit], f[n-1], 0], dtype=np.float32)
					else: # merge to isplit, and jsplit was last source so set it to 0
						px = np.array([frac*x[isplit]+(1-frac)*y[jsplit], 0], dtype=np.float32)
						py = np.array([frac*y[isplit]+(1-frac)*y[jsplit], 0], dtype=np.float32)
						pf = np.array([f[isplit] + f[jsplit], 0], dtype=np.float32)
					low_n = n-1
					pn = n-1
					goodmove = True # merge will be within image, and above min flux
			if goodmove:
				factor = np.log(truealpha-1) + (truealpha-1)*np.log(trueminf) - truealpha*np.log(frac*(1-frac)*sum_f) + 2*np.log(2*kickrange) - np.log(imsz[0]*imsz[1]) + np.log(low_n*(low_n+1)) - np.log(pairs) + np.log(sum_f) # last term is Jacobian
				factor *= (pn - n)
			nw = 2
		nmov[i] = nw
		dt1[j] += time.clock() - t1

		t2  = time.clock()
		if goodmove:
			dmodel, diff2 = image_model_eval(np.concatenate((px, np.extract(mover, x))), np.concatenate((py, np.extract(mover, y))), np.concatenate((pf, -np.extract(mover, f))), dback, imsz, cff, ref=resid, lib=libmmult.pcat_model_eval)

			plogL = -0.5*diff2
			if np.log(np.random.uniform()) < plogL + factor - logL -1.5*(pn-n):
				np.place(x, mover, px)
				np.place(y, mover, py)
				np.place(f, mover, pf)
				n = pn
				back += dback
				model += dmodel
				resid -= dmodel
				logL = plogL
				acceptance[j] += 1
		else:
			acceptance[j] += 1 # null move always accepted
		dt2[j] += time.clock() - t2
		
		if i == 0:
			plt.clf()
			plt.subplot(1,3,1)
			plt.imshow(mock, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(mock), vmax=np.percentile(mock, 95))
			plt.scatter(truex, truey, marker='+', s=np.sqrt(truef), color='g')
			plt.scatter(x[0:n], y[0:n], marker='x', s=np.sqrt(f[0:n]), color='r')
			plt.xlim(-0.5, imsz[0]-0.5)
			plt.ylim(-0.5, imsz[1]-0.5)
			plt.subplot(1,3,2)
			plt.imshow(resid, origin='lower', interpolation='none', cmap='bwr', vmin=-40, vmax=40)
			if j == 0:
				plt.tight_layout()
			plt.subplot(1,3,3)
			plt.hist(np.log10(truef), range=(np.log10(np.min(truef)), np.log10(np.max(truef))), log=True, alpha=0.5, label='Mock')
			plt.hist(np.log10(f[0:n]), range=(np.log10(np.min(truef)), np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain')
			plt.legend()
			plt.xlabel('log10 flux')
			plt.ylim((0.5, nstar))
			plt.draw()
			plt.pause(1e-5)
	xsample[j,:] = x
	ysample[j,:] = y
	fsample[j,:] = f
	acceptance[j] /= float(nloop)
	print 'Loop', j, 'Acceptance', acceptance[j], 'background', back, 'n', n, 'dt1', dt1[j], 'dt2', dt2[j], 'nmov', np.mean(nmov)

print 'dt1 avg', np.mean(dt1), 'dt2 avg', np.mean(dt2)
