program perceptron
    implicit none
    real :: input(4, 3)  ! Training data --- 4 by 3 matrix
    real :: output(4, 1) ! Outputs for training data 
    real :: w(3, 1)      ! Weight vector
    real :: alpha        ! Learning rate 
    real :: start, end   ! Start/end times for program duration
    integer :: i         ! Iterator variable

    ! Initialize training input values
    data input / 0, 0, 1, &
                 1, 1, 1, & 
                 1, 0, 1, &
                 0, 1, 1 / 

    ! Initialize training output values
    data output / 0, 1, 1, 0 / 

    ! Initialize weights with random numbers in [0, 1)
    call random_number(w)

    call cpu_time(start)

    ! ... 

    call cpu_time(end)

end program perceptron

! **********************************************************
! sigmoid activation function
! Returns the derivative of sigmoid if is_derivative is .true.
subroutine sigmoid(x, is_derivative)
    implicit none
    real, intent(inout) :: x 
    logical, intent(in) :: is_derivative
    real :: sig 

    sig = 1 / (1 + exp(-x))

    if (is_derivative) then
        x = sig * (1 - sig)
    else
        x = sig
    end if
end subroutine sigmoid 
! ********************************************************** 
! tanh activation function
subroutine tanhaf(x, is_derivative)
    implicit none 
    real, intent(inout) :: x
    logical, intent(in) :: is_derivative 
    real :: t

    t = tanh(x)

    if (is_derivative) then
        x = 1 - t**2
    else 
        x = t
    end if
end subroutine tanhaf
! **********************************************************
