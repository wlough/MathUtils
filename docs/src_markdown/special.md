

## `special` submodule

### Legendre polynomials

The Legendre polynomials are a set of polynomials which form an orthogonal basis for \f$L^2([-1,1])\f$. The Legendre polynomial of degree \f$n\f$, for \f$n=0,1,2,\ldots\f$, is denoted \f$P_{n}\f$.

__Differential equation.__

$$
(1-x^2)\frac{\mathrm{d}^2}{\mathrm{d}x^2}P_n(x)
-2x\frac{\mathrm{d}}{\mathrm{d}x}P_n(x)
+n(n+1)P_n(x)
=
0
$$


### Associated Legendre polynomials

__Standard normalization.__

The associated Legendre 'polynomials' are a set of functions belonging to \f$L^2([-1,1])\f$. The associated Legendre polynomial of degree \f$\ell\f$ and order \f$m\f$, for \f$\ell=0,1,2,\ldots\f$ and \f$m=-\ell,\ldots,\ell\f$, is denoted \f$P_{\ell}^{m}\f$. Technically, \f$P_{\ell}^{m}\f$ is only a polynomial when \f$m\f$ is even. In this section our definition of the associated Legendre polynomials includes the Condon–Shortley phase.

__Differential equation.__




### Spherical harmonics

__Standard complex spherical harmonics.__
Spherical harmonics are a set of functions defined on the unit sphere \f$S^2\f$ which are orthonormal to the \f$L^2(S^2)\f$ inner product. The spherical harmonic of degree \f$\ell\f$ and order \f$m\f$, for \f$\ell=0,1,2,\ldots\f$ and \f$m=-\ell,\ldots,\ell\f$, is denoted \f$\Ylm{\ell}{m}\f$.

__Orthonormality.__
\f[
\int_{S^2}
\Ylm{\ell}{m}^{*}
\Ylm{\ell'}{m'}
\dd{A}_{S^2}
=
\delta_{\ell\ell'}\delta_{mm'}
\f]

__Symmetries.__

\f[
\Ylm{\ell}{m}^{*}
=
(-1)^m
\Ylm{\ell}{-m}
\f]
\f[
\Ylm{\ell}{m}(\pi-\theta, \pi)
=
(-1)^{\ell-|m|}
\Ylm{\ell}{m}(\theta, \pi)
\f]

__Relation to associated Legendre polynomials.__

\f[
\Ylm{\ell}{m}(\theta, \phi)
=
\sqrt{\frac{(2\ell+1)(\ell-m)!}{4\pi (\ell+m)!}}
e^{im\phi}
\Plm{\ell}{m}(\cos\theta)
=
e^{im\phi}
\sPlm{\ell}{m}(\theta)
\f]


__Trig series representation.__

\f[
Y_{\ell}^{m}(\theta,\phi)
=
i^{m+|m|}
e^{i m\phi}
\sum_{k=0}^{\lfloor(\ell-|m|)/2\rfloor}
(-1)^k
N_{k\ell m}
\cos^{\ell-(|m|+2k)}(\theta)
\sin^{|m|+2k}(\theta)    
\f]
where the \f$N_{k\ell m}\f$'s are given by
\f[
N_{k\ell m}
=
\frac{\sqrt{
(2\ell+1)(\ell+m)!(\ell-m)!
}}{
\sqrt{4\pi}
2^{|m|+2k}
(\ell-|m|-2k)!
(|m|+k)!
k!
}
\f]
Note that \f$m+|m|=0\f$ whenever \f$m\leq 0\f$ and \f$m+|m|=2m\f$ when \f$m>0\f$, so
\f[
i^{m+|m|}
=
\begin{cases}
-1 & m > 0 \text{ and } m \text{ is odd}\\
1 & \text{otherwise }
\end{cases}
\f]



### Real spherical harmonics

\begin{gather}
\rYlm{\ell}{m}(\theta,\phi)
=
\begin{cases}
\sqrt{2}(-1)^{m}Im[\Ylm{\ell}{|m|}(\theta,\phi)] & m<0\\
\Ylm{\ell}{0}(\theta,\phi) & m=0\\
\sqrt{2}(-1)^{m}Re[\Ylm{\ell}{m}(\theta,\phi)] & m>0\\
\end{cases}
\end{gather}


__Orthonormality.__
\f[
\int_{S^2}
\rYlm{\ell}{m}
\rYlm{\ell'}{m'}
\dd{A}_{S^2}
=
\delta_{\ell\ell'}\delta_{mm'}
\f]



__Relation to associated Legendre polynomials.__

\f[
\begin{aligned}
\rYlm{\ell}{m}(\theta, \phi)
&=
\begin{cases}
\sqrt{2}(-1)^{m}\sin(|m|\phi)\sPlm{\ell}{|m|}(\theta) & m<0\\
\sPlm{\ell}{0}(\theta) & m=0\\
\sqrt{2}(-1)^{m}\cos(im\phi)\sPlm{\ell}{m}(\theta) & m>0\\
\end{cases}
\\
&=
\sPlm{\ell}{m}(\theta)
\begin{cases}
-\sqrt{2}\sin(m\phi) & m<0\\
1 & m=0\\
(-1)^{m}\sqrt{2}\cos(m\phi) & m>0\\
\end{cases}
\end{aligned}
\f]



