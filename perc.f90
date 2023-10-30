program perceptron
    implicit none
    real :: oi(4, 3)      ! Training data input --- 4 by 3 matrix
    real :: tj(4, 1)      ! Target output vector for training data 
    real :: w(3, 1)       ! Weight vector
    real :: oj(4, 1)      ! Second layer of NN (output)
    real :: delta(4, 1)   ! Delta to update weights 
    real :: error(4, 1)   ! Error vector

    real :: alpha(4)      ! Learning rates
    real :: start, end    ! Start/end times for program duration
    integer :: i          ! Iterator variable

    integer :: io         ! File input/output
    character(32) :: filename ! Filename 

    ! Initialize training input values
    data oi / 0, 0, 1, &
              1, 1, 1, & 
              1, 0, 1, &
              0, 1, 1 / 

    ! Initialize training output values
    data tj / 0, 1, 1, 0 / 

    ! Initialize weights with random numbers in [-1, 1)
    call random_seed() 


    ! Initialize training rates
    alpha = [0.1, 1.0, 10.0, 100.0]

    ! call cpu_time(start)

    do io = 1, 4
        ! Construct the filename
        write(filename, '(A,F6.1,A)') "alpha", alpha(io), ".csv"
        open(unit=io, file=filename, status='replace', action='write')
        ! CSV header
        write(io,*)"epoch,error"

        ! Randomize weights
        call random_number(w)
        w = (2 * w) - 1

        do i = 1, 100000
            ! Forward propagation with sigmoid transfer function
            oj = 1. / (1. + exp(-matmul(oi, w)))
            ! oj = tanh(matmul(oi, w))

            error = oj - tj
            ! Find delta (amount to update weights)
            delta = error * (oj * (1. - oj))
            ! delta = error * (1 - tanh(oj)**2)

            w = w - alpha(io) * matmul(transpose(oi), delta)
            ! Calculate vector norm of error and write to file
            write(io,*)i,",",norm2(error)
        end do
        close(io)

        print *, oj
    end do

    ! call cpu_time(end)

end program perceptron