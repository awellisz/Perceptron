program perceptron
    implicit none
    real :: oi(4, 3)      ! Training data input --- 4 by 3 matrix
    real :: tj(4, 1)      ! Target output vector for training data 
    real :: w(3, 1)       ! Weight vector
    real :: oj(4, 1)      ! Second layer of NN (output)
    real :: delta(4, 1)   ! Delta to update weights 
    real :: alpha         ! Learning rate 
    real :: start, end    ! Start/end times for program duration
    integer :: i          ! Iterator variable

    ! Initialize training input values
    data oi / 0, 0, 1, &
              1, 1, 1, & 
              1, 0, 1, &
              0, 1, 1 / 

    ! Initialize training output values
    data tj / 0, 1, 1, 0 / 

    ! Initialize weights with random numbers in [-1, 1]
    call random_number(w)
    w = (2 * w) - 1

    ! Initialize training rate
    alpha = 1

    call cpu_time(start)

    do i = 1, 100000
        ! Forward propagation with sigmoid transfer function
        oj = 1. / (1. + exp(-matmul(oi, w)))
        ! oj = tanh(matmul(oi, w))

        ! Find delta (amount to update weights)
        delta = (oj - tj) * (oj * (1. - oj))
        ! delta = (oj - toutput) * (1 - tanh(oj)**2)

        w = w - alpha * matmul(transpose(oi), delta)
    end do

    call cpu_time(end)

    print *, oj

end program perceptron