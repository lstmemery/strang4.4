### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ f371bdae-9b1c-11eb-0d7e-6b6f1b89050b
using LinearAlgebra

# ╔═╡ 8fe4fb68-39ba-4ff6-82a7-5078482d4121
using Random

# ╔═╡ 8b96556a-28eb-455f-8011-785b0c921ed0
using RowEchelon

# ╔═╡ 7be7a27a-fe4a-49b5-b874-dd2c02ea836f
are_orthonormal(a, b) = (round(sum(a .^ 2)) == 1) & (round(sum(b .^ 2)) == 1) & (round((a'*b)[1]) == 0)

# ╔═╡ 51e478ef-7ea6-41b6-9478-2b7a7dcf75f0
normalize(a) = a / sqrt(sum(a .^ 2))

# ╔═╡ c1763f30-bb77-4371-aa02-2c573f18fccd
rat(a) = rationalize(a, tol=1e-4)

# ╔═╡ 53e17051-9217-4283-806c-5070f6a294dc
p(A, b) = A * inv(A' * A) * A' * A

# ╔═╡ b8a1693f-824a-4986-ae89-4053cb63a294
e(a, b) = b - p(a,b)

# ╔═╡ b6f6ca5e-0379-4a64-a908-19a0d620d450
function gram_schmidt(A)
	R = zeros(Float64, size(A, 1), size(A, 2))
	Q = zeros(Float64, size(A, 1), size(A, 2))
	for column_index in 1:size(A, 2)
		v = A[:, column_index]
		for row_index in 1:(column_index - 1)
			R[row_index, column_index] = Q[:,row_index]' * v
			v = v - R[row_index, column_index] * Q[:, row_index]
		end
		R[column_index, column_index] = norm(v)
		Q[:, column_index] .= v/R[column_index, column_index]
	end
	return Q, R
end 

# ╔═╡ 23eb402c-b85e-40d2-b496-f938960c23cd
Q(A) = gram_schmidt(A)[1]

# ╔═╡ a4e01889-2a98-4e91-90b0-5eebb387348d
X_hat(A, b) = inv((A'*A)) * A' * b

# ╔═╡ 9115072b-4c7b-45ae-a8e9-52c6a339d295
reflect_matrix(u) = I - u*u'

# ╔═╡ 809ab108-bdb7-4207-bc44-6bcb9a61bbf0
md"""
``\newcommand\m[1]{\begin{bmatrix}#1\end{bmatrix}}``
"""

# ╔═╡ aa9463f3-cc73-4e0f-aa39-09ac0ff498f7
ex_1_A = [
	1 2 3;
	-1 0 -3;
	0 -2 3
]

# ╔═╡ 397e5839-1cf5-4a26-b9b1-b8545f390df8
ex_1_Q, ex_1_R = gram_schmidt(ex_1_A)

# ╔═╡ 023d9b92-af51-44fd-bbc9-a8bd76c3f1b3
ex_1_Q .^ 2

# ╔═╡ 623d8190-1de4-4247-b272-67542556379d
md"""
## Q1
"""

# ╔═╡ 314a4fe6-c481-4cbf-aa0b-a6bd25bb813f
q1aa = [1 0]'

# ╔═╡ cda9eab8-3a78-403a-82a9-b00f05a759a1
q1ab = [-1 1]'

# ╔═╡ 5a14fb67-4d5e-40b4-92a6-9d2a8dbe69f3
are_orthonormal(q1aa, q1ab)

# ╔═╡ 226e020c-5fbc-4650-9b69-2ea12b8a34fd
q1aa' * q1ab

# ╔═╡ 5e1d91b9-94d6-460c-8377-904b49a59796
sum(q1ab .^ 2)

# ╔═╡ 4851865c-8fc7-44e1-94b3-631d50557d0e
are_orthonormal(q1aa, [0 1]')

# ╔═╡ 54bae67d-6bb5-4b9c-b6fc-b1f59f2e5f65
q1ba = [0.6 0.8]'

# ╔═╡ cce94808-2c17-4e4e-a630-9fe2bbf4fbc2
q1bb = [0.4 -0.3]'

# ╔═╡ 8be280e3-9b01-4de1-b963-2240dfd1c1eb
are_orthonormal(q1ba, q1bb)

# ╔═╡ 62d7d203-66d0-48dc-998b-72b7ab97557c
round.(q1ba' * q1bb)

# ╔═╡ c1e53c7f-2f8f-4199-be05-5bf6585787c7
are_orthonormal(q1ba, q1bb .* 2)

# ╔═╡ df232a91-68c5-4822-a8c4-d260cca69026
q1theta = rand()

# ╔═╡ 3af24a7e-9595-490f-8ed7-537c20eab801
q1ca = [cos(q1theta) sin(q1theta)]'

# ╔═╡ b4e6f936-c6af-4fa3-910e-7491a61c6bc9
q1cb = [-sin(q1theta) cos(q1theta)]'

# ╔═╡ 57e49431-dcb2-4c9d-9f28-1c8bf63311b4
are_orthonormal(q1ca, q1cb)

# ╔═╡ bac6fd60-f2f0-46d4-81de-f8870f048aa4
md"""
## Q2
"""

# ╔═╡ 2b3664fe-27ea-4609-8ab5-3ba9dc9ce969
q2a = [2 2 -1]'

# ╔═╡ 63681856-4c74-4121-9f61-a5f7c962d8d1
q2b = [-1 2 2]'

# ╔═╡ 60d879e4-fc5e-484b-affd-23ea5dedc047
q2q1 = normalize(q2a)

# ╔═╡ 692bf3b2-1908-400d-a993-170e29879d34
q2q2 = normalize(q2b)

# ╔═╡ 595433f3-9be3-4ce7-9558-50ec76669948
q2Q = cat(q2q1, q2q2, dims=2)

# ╔═╡ 7e82ddf1-cdb1-4b9e-a3a0-ab35956f395c
round.(q2Q' * q2Q, digits=5)

# ╔═╡ 23969da8-8a72-4af2-867a-45d50a757aa0
rat.(q2Q * q2Q')

# ╔═╡ 8d726357-3b77-40da-a74d-ad5daa2d9acc
md"""
## Q3
"""

# ╔═╡ 3dfadc1c-05ee-4dd3-88c6-12f9fd8ccb41
q3aA = I(3) * 4

# ╔═╡ 251cde85-18db-492e-8292-def0e5d35245
q3aA' * q3aA

# ╔═╡ 44870942-d391-4555-b60b-618717e276a3
q3bA = diagm([1,2,3])

# ╔═╡ df1e4cf0-4f99-4756-992d-158541248bee
q3bA' * q3bA

# ╔═╡ 7e89149e-14f2-46d4-963a-b4398d81edbd
md"""
## Q4
"""

# ╔═╡ 448946f6-deee-4341-86cb-cb5f0a603d78
q4a = [
	1 0;
	0 1;
	0 0
]

# ╔═╡ dcf2d046-9296-41ab-a2cf-ceb60a064dde
q4a * q4a'

# ╔═╡ 9f7d034e-f38e-4c5c-937f-145126cd9bb7
are_orthonormal(q4a[:,1], q4a[:,2])

# ╔═╡ 5ff9f0e5-be08-4b74-9947-7d9f583d37c7
md"""
b) A matrix made of ``0`` vectors would be orthogonal but not indepedent
"""

# ╔═╡ 6eea7077-3485-4764-a7e4-2faaf54dec03
q4cq1 = [1 1 1]' / sqrt(3)

# ╔═╡ 983dc14f-5b85-4a21-9261-ddfd758fe4bb
q4cq2 = [1 1 -2]' / sqrt(6)

# ╔═╡ 67c1af82-6dac-4a69-8009-1e4725e42681
sum(q4cq2 .^ 2)

# ╔═╡ 2716f87a-ec5f-46e2-961d-8a837000c34c
q4cq1' * q4cq2

# ╔═╡ f4b6a0ed-c1fb-48af-9073-30ddf2976875
q4cq3 = [1 -1 0]' / sqrt(2)

# ╔═╡ eddf9454-1549-4698-a18d-3156e7ff7528
q4cq3' * q4cq2

# ╔═╡ 664ff1e9-5c8c-4b34-aded-08d96414d14f
q4cq3' * q4cq1

# ╔═╡ 17f5357f-45f6-4293-8f51-4f762df86c4a
md"""
## Q5
"""

# ╔═╡ b432c562-bfa2-43a0-b1e5-3ecd0174c881
q5A = [1 1 2]

# ╔═╡ 88a8afc2-2adf-437c-89ef-952aa89df24f
q5v1 = [-1 1 0]'

# ╔═╡ 6cc9eea0-a245-46f7-8ab4-f5890eeffbe1
q5v2 = [-2 0 1]'

# ╔═╡ 17ad49d3-ad30-4838-ae88-e915319fdcde
q5Q = Q(cat(q5v1, q5v2, dims=2))

# ╔═╡ a25b4788-02d5-4f75-aaf2-ba831e33017e
q5Q[:,1]' * q5Q[:,2]

# ╔═╡ 9278c43a-a249-4584-9668-a4802f8c93b9
md"""
## Q6

``Q_1^TQ_1 = I``

``Q_2^TQ_2 = I``

``(Q_1Q_2)^T(Q_1Q_2) = Q_2^TQ_1^TQ_1Q_2 = Q_2^TIQ_2 = I``
"""

# ╔═╡ 4cfacba0-48e7-4529-9cfa-7194bce00f98
md"""
## Q7

Normal equation:

``A^TA\hat{x} = A^Tb``

When Orthogonal:

``Q^TQ\hat{x} = Q^Tb = \hat{x}``
"""

# ╔═╡ b464571a-3ad0-42b3-a5cb-4c51011b61c7
md"""
## Q8

``(q_1b)q_1 + (q_2b)q_2``
"""

# ╔═╡ 70aaf69d-3646-4a4b-baec-db3952bedbd7
md"""
## Q9
"""

# ╔═╡ c177f7a7-cbac-413b-a34e-68baa596aa63
q9Q = [
	0.8 -0.6;
	0.6 0.8;
	0 0
]

# ╔═╡ bb924963-686b-48a7-9d39-4e521ae167fb
q9P = q9Q * q9Q'

# ╔═╡ aa30a6be-20e3-4419-8ead-f1033ced87a4
q9P^2

# ╔═╡ 258ef7b7-b613-43ef-9e8e-338c0841c465
md"""
``(QQ^T)^2 = (QQ^T)^TQQ^T = QQ^TQQ^T = QQ^T``
"""

# ╔═╡ f03939e0-efc5-485e-8980-9b238f50324a
md"""
## Q10

a)``c_1=0`` when ``q_2^Tq_3``

``c_2=0`` when ``q_1^Tq_3``

``c_3=0`` when ``q_2^Tq_1``


b) 

``Q^TQx=0 \rightarrow x=0``
"""

# ╔═╡ 253222b0-7cf0-45c9-af26-9f1d1ce00fc0
md"""
## Q11
"""

# ╔═╡ b71c6f5d-e302-43c4-92a8-2ac3c31bb08e
q11A = [
	1 -6;
	3 6;
	4 8;
	5 0;
	7 8
]

# ╔═╡ 01bcf6eb-04cc-41da-a7f3-b2ff8ca18bb7
q11Q = Q(q11A)

# ╔═╡ 72a7dcb6-65fd-4420-a4a8-576a2b6b5eda
q11x_hat = q11Q' * [1 0 0 0 0]'

# ╔═╡ 4271c01e-f4e2-4610-8ea9-443f3baa3f2d
rat.(q11Q * q11x_hat)

# ╔═╡ a1643a29-ead3-42e3-847d-ac832fe197b2
md"""
## Q12

``x_1a_1 + x_2a_2 + x_3a_3 = b``

a) ``x_1a_1^Ta_1 + x_2a_1^Ta_2 + x_3a_1^Ta_3 = a_1^Tb``

``x_1 = a_1^Tb``

b) ``x_1a_1^Ta_1 + x_2a_1^Ta_2 + x_3a_1^Ta_3 = a_1^Tb``

``x_1a_1^Ta_1 = a_1^Tb``

``x_1 = \frac{a_1^Tb}{a_1^Ta_1}``

c) ``A^{-1}b``
"""

# ╔═╡ 9441646c-2fda-472e-a817-04e11396861b
md"""
## Q13
"""

# ╔═╡ 157e6187-a3dd-48fd-916e-b21fa5acc4ad
q13a = [1 1]'

# ╔═╡ 127f712d-1f2d-44ef-90fb-73b60478afb0
q13b = [4 0]'

# ╔═╡ 7454c1bd-b982-4af9-8d46-9f73967c7da8
q13p = (q13a' * q13b) / (q13a' * q13a)

# ╔═╡ fd54c1ad-1a38-48ce-9ef8-94813b228b86
q13B = q13b - q13p[1] * q13a

# ╔═╡ 795c13fe-19b7-4f87-9ffe-9bfa444d3d79
md"""
## Q14
"""

# ╔═╡ e0b3261b-6728-4c7d-a559-b3deafe21e6c
q14A = [
	1 4;
	1 0
]

# ╔═╡ 08cadb77-48f0-49c2-b477-4e1060eb01b5
gram_schmidt(q14A)

# ╔═╡ 599534fd-dde7-43d7-a149-5537327dfeef
md"""
## Q15
"""

# ╔═╡ 1ff66ec0-d0c3-4264-a2d8-cb74cb2bf4d4
q15A = [
	1 1;
	2 -1;
	-2 4
]

# ╔═╡ 3a3412b8-3b75-4f44-9c7f-7738b203356d
q15Q = Q(q15A)

# ╔═╡ 08da3718-4432-486a-8c2b-bc208ef36eb1
q15q3 = [-2/3 2/3 1/3]'

# ╔═╡ 6065b704-c2bb-4377-8221-69a6e4437dee
q15Q' * q15q3

# b) This third vector is the in the left nullspace

# ╔═╡ e9d0a0b6-63b2-4b79-af96-cd84d9e206c9
q15Q' * [1 2 7]'

# ╔═╡ 7218d98f-884b-4179-9d8d-0bf9fe1bcfd3
X_hat(q15A, [1 2 7]')

# ╔═╡ 108df804-275d-47d6-9037-dd0d1faf389e
X_hat(q15Q, [1 2 7]') # These are not the same!

# ╔═╡ 24860526-b768-4b80-929c-cec1caa94207
md"""
## Q16
"""

# ╔═╡ 950a8699-4342-476e-9b64-3696d393e476
q16a = [4 5 2 2]'

# ╔═╡ 85cab0b1-3ec2-4876-b3e3-9242bd126000
q16b = [1 2 0 0]'

# ╔═╡ eff686ff-6aab-4856-85ff-67c20cfc2960
rat.((q16a' * q16b) / (q16a' * q16a))

# ╔═╡ 2c5bc7d7-3c1d-4e9d-8a2b-cf7ef480bdf9
rat.(Q(cat(q16a, q16b, dims=2)))

# ╔═╡ 46d6cc98-bec6-436b-afb8-2f0f94196835
md"""
## Q17
"""

# ╔═╡ 9c50709b-4fe6-43eb-9d4e-ef9698710d17
q17a = [1 1 1]'

# ╔═╡ 18c7be17-2303-43cc-acd4-9e06c0a36898
q17b = [1 3 5]'

# ╔═╡ 7c35a217-e40b-4347-96b6-08252e71dd01
q17p = p(q17a, q17b)

# ╔═╡ 7d533ad2-8319-4257-9ad2-47a75a2703b4
q17e = e(q17a, q17b)

# ╔═╡ 766ab627-d2f8-403d-9d09-2198c32f3b19
q17q1 = normalize(q17a)

# ╔═╡ b634f474-aeb4-4d50-87ff-8e31721e62ca
q17q1 / (1/sqrt(3))

# ╔═╡ 44062f26-790b-4a7c-8600-d77471ccbb21
q17q2 = normalize(q17e)

# ╔═╡ 31a02f40-5c68-440c-baa5-90e6b91373bb
q17q2 / (1 / sqrt(2))

# ╔═╡ f90d3b6f-d3df-47a6-b28f-8cdf47fcc8f3
Q(cat(q17a, q17b, dims=2))

# ╔═╡ afe22002-9e34-418d-99aa-11be70b1d3f3
md"""
## Q18
"""

# ╔═╡ 3375c356-159c-44d5-83d4-6c847dd8074f
q18A = [
	1 -1 0 0;
	0 1 -1 0;
	0 0 1 -1
]'

# ╔═╡ 4b85ad34-8af8-43c4-958d-6bbda00d2025
q18Q = Q(q18A)

# ╔═╡ a4c74bb4-f708-4922-bfa7-84cc23b42ae8
q18Q[:, 1] * sqrt(2)

# ╔═╡ f8a443f6-a72f-4eb9-8c0e-0233d4c9e95c
q18Q[:, 2] * sqrt(6)

# ╔═╡ 4fad2526-a1bf-444a-a218-1d9c09a85bf8
q18Q[:, 3] * sqrt(3)

# ╔═╡ 8fdaf8b9-a768-4bce-9f1d-eb83d420de58
md"""
## Q19

Lower triangular by upper triangular
"""

# ╔═╡ 81407217-2886-4602-ab74-a80bf9406ab5
q19A = [
	-1 1;
	2 1;
	2 4
]

# ╔═╡ 0065e06b-e5a1-4485-a191-9b411921db6a
gram_schmidt(q19A)

# ╔═╡ 7bc5d220-403a-4336-a023-c5d79b2d61e0
md"""
## Q20
"""

# ╔═╡ 87412e15-2d27-4132-8da6-58df6e22a235
q20Q = [
	0 1;
	1 0
]

# ╔═╡ 33d3ee8f-df1e-417d-aa4e-155ffa9d0c2a
Q(q20Q)

# ╔═╡ 8f35f9f2-7da6-4bb8-8a3d-37cb4f3dca49
Q(q20Q\I)

# ╔═╡ fff7b10e-313b-4faf-acb8-7574a87cf5e7
q20Ab = [
	1 1;
	-7 9;
	11 15
]

# ╔═╡ b5592430-075c-49c4-ab83-05be997605a4
q20Qb = Q(q20Ab)

# ╔═╡ ebf25c41-f28b-4e4d-876e-54e3f19355e9
norm(q20Qb * [100, 1])

# ╔═╡ d38fb04c-3529-4918-92a5-6e65da29cb7f
norm([100, 1])

# ╔═╡ 58ce79a3-6a5c-426f-9c75-ae1aa46ce016
md"""
## Q21
"""

# ╔═╡ f3127687-f5e7-4bed-ab10-9d60381e55f4
q21A = [
	1 -2;
	1 0;
	1 1;
	1 3
]

# ╔═╡ 789c26fc-3431-4c8d-b776-92adf8798ab4
q21Q = Q(q21A)

# ╔═╡ ebd4dd6c-b174-49c7-aeae-4e5571c6b602
q21b = [-4 -3 3 0]'

# ╔═╡ 965d7fac-c4da-4000-9082-3da7364558e2
q21_x_hat = q21Q' * q21b

# ╔═╡ a3ab7616-e24d-444d-94a8-fb372086e3e0
rat.(q21Q * q21_x_hat)

# ╔═╡ 791073f9-0881-4a5c-b46e-3906a3e42ad8
md"""
## Q22

Note, we are looking for orthogonal, not orthonormal vectors
"""

# ╔═╡ a8697d60-55d7-4704-b3f1-75850d19d199
q22A = [
	1 1 1;
	1 -1 0;
	2 0 4
]

# ╔═╡ 654e3cab-060d-40a2-8715-740e905f91e5
q22Q, q22R = gram_schmidt(q22A)

# ╔═╡ 5b2ffe45-9224-4f89-9542-03bd93fa151d
q22Q * diagm(1 .\ (diag(q22R)))

# ╔═╡ db06db7b-11ef-4a65-be9d-efa51703074b
md"""
## Q23
"""

# ╔═╡ b4b52546-5526-44cb-846d-5a35ae9cf3f3
q23A = [
	1 2 4;
	0 0 5;
	0 3 6
]

# ╔═╡ b391c2ca-8d26-4e21-a3f5-db928a9b6041
q23Q, q23R = gram_schmidt(q23A)

# ╔═╡ fc9fce7a-3157-4224-8186-9965af8f5079
q23Q * diagm(1 .\ (diag(q23R)))

# ╔═╡ a955c6c5-f38f-43b2-8105-83974fc3d4cf
md"""
## Q24

a)

``A = \m{1 & 1 & 1 & -1}``

basis: ``\m{-1 & 1 & 0 & 0 }, \m{-1 & 0 & 1 & 0 }, \m{1 & 0 & 0 & 1}``
"""

# ╔═╡ b29c0186-68ea-443a-87a0-e9de56773524
q24A = [
	-1 1 0 0;
	-1 0 1 0;
	1 0 0 1
]

# ╔═╡ a4021269-af5c-4eb7-bc60-7031df7fc3a5
rref(q24A)

# ╔═╡ f99c00a3-457c-4217-ba2c-41eaf41a9482
q24A * [-1, -1, -1, 1]

# ╔═╡ d5a5d6db-b905-40c7-941f-f99d6af2d3ca
q24null = [-1, -1, -1, 1]

# ╔═╡ 67c8f24a-270a-4b4d-8d27-5ee9af35bd65
md"""
## Q25
"""

# ╔═╡ fbf4e65d-2935-4543-a3b0-81c3ba8b6a15
q25A1 = [
	2 1;
	1 1
]

# ╔═╡ 23fc22fe-1a04-4e71-8ab7-af57e9008e1e
q25A2 = [
	1 1;
	1 1
]

# ╔═╡ 1723e34d-7521-4352-b31f-96524c0280b3
md"""
## Q26
"""

# ╔═╡ 0b58b09f-d473-4692-acec-856aa8718d6b
md"""
## Q27
"""

# ╔═╡ c28b341d-e66a-4a1a-8310-326fa02fe4bf
md"""
## Q28
"""

# ╔═╡ 6a0a3e20-e8e8-4b31-a0ff-fe0bfc9af437
md"""
## Q29
"""

# ╔═╡ 39457a85-2577-4e36-a943-fd90b8c10972
q29A = [
	2 0 1;
	2 -3 0;
	-1 3 0
]

# ╔═╡ b522edb7-3894-4a9e-87dd-368691fc11a6
qr(q29A)

# ╔═╡ 6b6e0f96-8e19-4c57-87f6-bfd4f1aa29ce
md"""
## Q30
"""

# ╔═╡ 304d0d2f-1d96-4ff6-ade0-4e726d8cf6b9
q30W = 0.5 * [
	1 1 sqrt(2) 0;
	1 1 -sqrt(2) 0;
	1 -1 0 sqrt(2);
	1 -1 0 -sqrt(2)
]

# ╔═╡ 09f03d55-79c5-40c8-9bbc-7997ad78c10b
md"""
## Q31
"""

# ╔═╡ 1db34a88-b8f9-449b-9a55-6062bc2b4930
q31Q = [
	1 -1 -1 -1;
	-1 1 -1 -1;
	-1 -1 1 -1;
	-1 -1 -1 1
]

# ╔═╡ f0966d05-4679-4597-ade8-f293882b31bf
q31b = [1 1 1 1]'

# ╔═╡ e09179dc-b572-47d0-a0c3-640128a9696d
q31c = 1

# ╔═╡ aa474d55-5cd7-4e30-95b5-49845ff16b6a
md"""
## Q32
"""

# ╔═╡ 410239ca-8383-46c8-aa9d-eabcc7101bfc
q32u1 = [0 1]'

# ╔═╡ e360234e-afb5-4666-9d7e-97672543aa3e
q32u2 = [0 sqrt(2)/2 sqrt(2)/2]'

# ╔═╡ c0f64216-6d45-45ea-a16c-03ba209e2b71
md"""
## Q33
"""

# ╔═╡ dffe6b56-2c3d-4ed4-92f5-eac080029f6a
md"""
## Q34
"""

# ╔═╡ 8e432ea5-175a-4152-beeb-5fff199bb6c9
md"""
## Q35
"""

# ╔═╡ cc72a4e9-f008-4adb-b0c9-63033cca4d80
q35A = diagm(0 => [1, 1, 1, 1], 1 => [-1, -1, -1])

# ╔═╡ 66e58f42-f3a8-4164-a2d6-040d15d451bc
qr(q35A)

# ╔═╡ b7c54e7c-367e-4d1d-bb73-83f8b402b226
md"""
## Q36
"""

# ╔═╡ 6ecf0a26-f7c7-412b-a8c7-4a082c6504d8
md"""
## Q37
"""

# ╔═╡ 8af9ace5-0d9e-4314-9d63-d1351844f6cc


# ╔═╡ Cell order:
# ╠═f371bdae-9b1c-11eb-0d7e-6b6f1b89050b
# ╠═8fe4fb68-39ba-4ff6-82a7-5078482d4121
# ╠═8b96556a-28eb-455f-8011-785b0c921ed0
# ╠═7be7a27a-fe4a-49b5-b874-dd2c02ea836f
# ╠═51e478ef-7ea6-41b6-9478-2b7a7dcf75f0
# ╠═c1763f30-bb77-4371-aa02-2c573f18fccd
# ╠═53e17051-9217-4283-806c-5070f6a294dc
# ╠═b8a1693f-824a-4986-ae89-4053cb63a294
# ╠═b6f6ca5e-0379-4a64-a908-19a0d620d450
# ╠═23eb402c-b85e-40d2-b496-f938960c23cd
# ╠═a4e01889-2a98-4e91-90b0-5eebb387348d
# ╠═9115072b-4c7b-45ae-a8e9-52c6a339d295
# ╠═809ab108-bdb7-4207-bc44-6bcb9a61bbf0
# ╠═aa9463f3-cc73-4e0f-aa39-09ac0ff498f7
# ╠═397e5839-1cf5-4a26-b9b1-b8545f390df8
# ╠═023d9b92-af51-44fd-bbc9-a8bd76c3f1b3
# ╠═623d8190-1de4-4247-b272-67542556379d
# ╠═314a4fe6-c481-4cbf-aa0b-a6bd25bb813f
# ╠═cda9eab8-3a78-403a-82a9-b00f05a759a1
# ╠═5a14fb67-4d5e-40b4-92a6-9d2a8dbe69f3
# ╠═226e020c-5fbc-4650-9b69-2ea12b8a34fd
# ╠═5e1d91b9-94d6-460c-8377-904b49a59796
# ╠═4851865c-8fc7-44e1-94b3-631d50557d0e
# ╠═54bae67d-6bb5-4b9c-b6fc-b1f59f2e5f65
# ╠═cce94808-2c17-4e4e-a630-9fe2bbf4fbc2
# ╠═8be280e3-9b01-4de1-b963-2240dfd1c1eb
# ╠═62d7d203-66d0-48dc-998b-72b7ab97557c
# ╠═c1e53c7f-2f8f-4199-be05-5bf6585787c7
# ╠═df232a91-68c5-4822-a8c4-d260cca69026
# ╠═3af24a7e-9595-490f-8ed7-537c20eab801
# ╠═b4e6f936-c6af-4fa3-910e-7491a61c6bc9
# ╠═57e49431-dcb2-4c9d-9f28-1c8bf63311b4
# ╠═bac6fd60-f2f0-46d4-81de-f8870f048aa4
# ╠═2b3664fe-27ea-4609-8ab5-3ba9dc9ce969
# ╠═63681856-4c74-4121-9f61-a5f7c962d8d1
# ╠═60d879e4-fc5e-484b-affd-23ea5dedc047
# ╠═692bf3b2-1908-400d-a993-170e29879d34
# ╠═595433f3-9be3-4ce7-9558-50ec76669948
# ╠═7e82ddf1-cdb1-4b9e-a3a0-ab35956f395c
# ╠═23969da8-8a72-4af2-867a-45d50a757aa0
# ╠═8d726357-3b77-40da-a74d-ad5daa2d9acc
# ╠═3dfadc1c-05ee-4dd3-88c6-12f9fd8ccb41
# ╠═251cde85-18db-492e-8292-def0e5d35245
# ╠═44870942-d391-4555-b60b-618717e276a3
# ╠═df1e4cf0-4f99-4756-992d-158541248bee
# ╠═7e89149e-14f2-46d4-963a-b4398d81edbd
# ╠═448946f6-deee-4341-86cb-cb5f0a603d78
# ╠═dcf2d046-9296-41ab-a2cf-ceb60a064dde
# ╠═9f7d034e-f38e-4c5c-937f-145126cd9bb7
# ╠═5ff9f0e5-be08-4b74-9947-7d9f583d37c7
# ╠═6eea7077-3485-4764-a7e4-2faaf54dec03
# ╠═983dc14f-5b85-4a21-9261-ddfd758fe4bb
# ╠═67c1af82-6dac-4a69-8009-1e4725e42681
# ╠═2716f87a-ec5f-46e2-961d-8a837000c34c
# ╠═f4b6a0ed-c1fb-48af-9073-30ddf2976875
# ╠═eddf9454-1549-4698-a18d-3156e7ff7528
# ╠═664ff1e9-5c8c-4b34-aded-08d96414d14f
# ╠═17f5357f-45f6-4293-8f51-4f762df86c4a
# ╠═b432c562-bfa2-43a0-b1e5-3ecd0174c881
# ╠═88a8afc2-2adf-437c-89ef-952aa89df24f
# ╠═6cc9eea0-a245-46f7-8ab4-f5890eeffbe1
# ╠═17ad49d3-ad30-4838-ae88-e915319fdcde
# ╠═a25b4788-02d5-4f75-aaf2-ba831e33017e
# ╠═9278c43a-a249-4584-9668-a4802f8c93b9
# ╠═4cfacba0-48e7-4529-9cfa-7194bce00f98
# ╠═b464571a-3ad0-42b3-a5cb-4c51011b61c7
# ╠═70aaf69d-3646-4a4b-baec-db3952bedbd7
# ╠═c177f7a7-cbac-413b-a34e-68baa596aa63
# ╠═bb924963-686b-48a7-9d39-4e521ae167fb
# ╠═aa30a6be-20e3-4419-8ead-f1033ced87a4
# ╠═258ef7b7-b613-43ef-9e8e-338c0841c465
# ╠═f03939e0-efc5-485e-8980-9b238f50324a
# ╠═253222b0-7cf0-45c9-af26-9f1d1ce00fc0
# ╠═b71c6f5d-e302-43c4-92a8-2ac3c31bb08e
# ╠═01bcf6eb-04cc-41da-a7f3-b2ff8ca18bb7
# ╠═72a7dcb6-65fd-4420-a4a8-576a2b6b5eda
# ╠═4271c01e-f4e2-4610-8ea9-443f3baa3f2d
# ╠═a1643a29-ead3-42e3-847d-ac832fe197b2
# ╠═9441646c-2fda-472e-a817-04e11396861b
# ╠═157e6187-a3dd-48fd-916e-b21fa5acc4ad
# ╠═127f712d-1f2d-44ef-90fb-73b60478afb0
# ╠═7454c1bd-b982-4af9-8d46-9f73967c7da8
# ╠═fd54c1ad-1a38-48ce-9ef8-94813b228b86
# ╠═795c13fe-19b7-4f87-9ffe-9bfa444d3d79
# ╠═e0b3261b-6728-4c7d-a559-b3deafe21e6c
# ╠═08cadb77-48f0-49c2-b477-4e1060eb01b5
# ╠═599534fd-dde7-43d7-a149-5537327dfeef
# ╠═1ff66ec0-d0c3-4264-a2d8-cb74cb2bf4d4
# ╠═3a3412b8-3b75-4f44-9c7f-7738b203356d
# ╠═08da3718-4432-486a-8c2b-bc208ef36eb1
# ╠═6065b704-c2bb-4377-8221-69a6e4437dee
# ╠═e9d0a0b6-63b2-4b79-af96-cd84d9e206c9
# ╠═7218d98f-884b-4179-9d8d-0bf9fe1bcfd3
# ╠═108df804-275d-47d6-9037-dd0d1faf389e
# ╠═24860526-b768-4b80-929c-cec1caa94207
# ╠═950a8699-4342-476e-9b64-3696d393e476
# ╠═85cab0b1-3ec2-4876-b3e3-9242bd126000
# ╠═eff686ff-6aab-4856-85ff-67c20cfc2960
# ╠═2c5bc7d7-3c1d-4e9d-8a2b-cf7ef480bdf9
# ╠═46d6cc98-bec6-436b-afb8-2f0f94196835
# ╠═9c50709b-4fe6-43eb-9d4e-ef9698710d17
# ╠═18c7be17-2303-43cc-acd4-9e06c0a36898
# ╠═7c35a217-e40b-4347-96b6-08252e71dd01
# ╠═7d533ad2-8319-4257-9ad2-47a75a2703b4
# ╠═766ab627-d2f8-403d-9d09-2198c32f3b19
# ╠═b634f474-aeb4-4d50-87ff-8e31721e62ca
# ╠═44062f26-790b-4a7c-8600-d77471ccbb21
# ╠═31a02f40-5c68-440c-baa5-90e6b91373bb
# ╠═f90d3b6f-d3df-47a6-b28f-8cdf47fcc8f3
# ╠═afe22002-9e34-418d-99aa-11be70b1d3f3
# ╠═3375c356-159c-44d5-83d4-6c847dd8074f
# ╠═4b85ad34-8af8-43c4-958d-6bbda00d2025
# ╠═a4c74bb4-f708-4922-bfa7-84cc23b42ae8
# ╠═f8a443f6-a72f-4eb9-8c0e-0233d4c9e95c
# ╠═4fad2526-a1bf-444a-a218-1d9c09a85bf8
# ╠═8fdaf8b9-a768-4bce-9f1d-eb83d420de58
# ╠═81407217-2886-4602-ab74-a80bf9406ab5
# ╠═0065e06b-e5a1-4485-a191-9b411921db6a
# ╠═7bc5d220-403a-4336-a023-c5d79b2d61e0
# ╠═87412e15-2d27-4132-8da6-58df6e22a235
# ╠═33d3ee8f-df1e-417d-aa4e-155ffa9d0c2a
# ╠═8f35f9f2-7da6-4bb8-8a3d-37cb4f3dca49
# ╠═fff7b10e-313b-4faf-acb8-7574a87cf5e7
# ╠═b5592430-075c-49c4-ab83-05be997605a4
# ╠═ebf25c41-f28b-4e4d-876e-54e3f19355e9
# ╠═d38fb04c-3529-4918-92a5-6e65da29cb7f
# ╠═58ce79a3-6a5c-426f-9c75-ae1aa46ce016
# ╠═f3127687-f5e7-4bed-ab10-9d60381e55f4
# ╠═789c26fc-3431-4c8d-b776-92adf8798ab4
# ╠═ebd4dd6c-b174-49c7-aeae-4e5571c6b602
# ╠═965d7fac-c4da-4000-9082-3da7364558e2
# ╠═a3ab7616-e24d-444d-94a8-fb372086e3e0
# ╠═791073f9-0881-4a5c-b46e-3906a3e42ad8
# ╠═a8697d60-55d7-4704-b3f1-75850d19d199
# ╠═654e3cab-060d-40a2-8715-740e905f91e5
# ╠═5b2ffe45-9224-4f89-9542-03bd93fa151d
# ╠═db06db7b-11ef-4a65-be9d-efa51703074b
# ╠═b4b52546-5526-44cb-846d-5a35ae9cf3f3
# ╠═b391c2ca-8d26-4e21-a3f5-db928a9b6041
# ╠═fc9fce7a-3157-4224-8186-9965af8f5079
# ╠═a955c6c5-f38f-43b2-8105-83974fc3d4cf
# ╠═b29c0186-68ea-443a-87a0-e9de56773524
# ╠═a4021269-af5c-4eb7-bc60-7031df7fc3a5
# ╠═f99c00a3-457c-4217-ba2c-41eaf41a9482
# ╠═d5a5d6db-b905-40c7-941f-f99d6af2d3ca
# ╠═67c8f24a-270a-4b4d-8d27-5ee9af35bd65
# ╠═fbf4e65d-2935-4543-a3b0-81c3ba8b6a15
# ╠═23fc22fe-1a04-4e71-8ab7-af57e9008e1e
# ╠═1723e34d-7521-4352-b31f-96524c0280b3
# ╠═0b58b09f-d473-4692-acec-856aa8718d6b
# ╠═c28b341d-e66a-4a1a-8310-326fa02fe4bf
# ╠═6a0a3e20-e8e8-4b31-a0ff-fe0bfc9af437
# ╠═39457a85-2577-4e36-a943-fd90b8c10972
# ╠═b522edb7-3894-4a9e-87dd-368691fc11a6
# ╠═6b6e0f96-8e19-4c57-87f6-bfd4f1aa29ce
# ╠═304d0d2f-1d96-4ff6-ade0-4e726d8cf6b9
# ╠═09f03d55-79c5-40c8-9bbc-7997ad78c10b
# ╠═1db34a88-b8f9-449b-9a55-6062bc2b4930
# ╠═f0966d05-4679-4597-ade8-f293882b31bf
# ╠═e09179dc-b572-47d0-a0c3-640128a9696d
# ╠═aa474d55-5cd7-4e30-95b5-49845ff16b6a
# ╠═410239ca-8383-46c8-aa9d-eabcc7101bfc
# ╠═e360234e-afb5-4666-9d7e-97672543aa3e
# ╠═c0f64216-6d45-45ea-a16c-03ba209e2b71
# ╠═dffe6b56-2c3d-4ed4-92f5-eac080029f6a
# ╠═8e432ea5-175a-4152-beeb-5fff199bb6c9
# ╠═cc72a4e9-f008-4adb-b0c9-63033cca4d80
# ╠═66e58f42-f3a8-4164-a2d6-040d15d451bc
# ╠═b7c54e7c-367e-4d1d-bb73-83f8b402b226
# ╠═6ecf0a26-f7c7-412b-a8c7-4a082c6504d8
# ╠═8af9ace5-0d9e-4314-9d63-d1351844f6cc
