import numpy as np
import random
import time
from scipy.optimize import linprog

class Simplex_Table(object):
    def __init__(self, ori_c, ori_A, ori_b, ori_res, var_type_arr, cons_type_arr, debug=False):  
        
        """ defined constant 
            NORMAL means var x
            NINF   means var x is replaced by -x
            POS    means var x is made from a pair (x+, x-)
        """       
        self.NINF = -1000000
        self.INF  =  1000000
        self.NORMAL = -1

        """ debug option """
        self.debug = debug

        """ numpy for std simplex method """
        self.std_c = ori_c.copy()
        self.std_A = ori_A.copy()
        self.std_b = ori_b.copy()

        self.ori_constrain_num = ori_A.shape[0]
        self.ori_var_num = ori_A.shape[1]
        self.var_type_arr = var_type_arr
        self.cons_type_arr = cons_type_arr
        self.var_status = np.full(self.ori_var_num,self.NORMAL)

        
        # change variable based on variable limitation
        for i in range(len(var_type_arr)):
            # if a variable has <= 0 limitation
            if (var_type_arr[i] == -1):
                ori_A[:,i] = -ori_A[:,i]
                ori_c[:,i] = -ori_c[:,i]
                self.var_status[i] = self.NINF
            # if a variable has no limitation
            if (var_type_arr[i] == 0):
                col_A = np.array(-ori_A[:,i]).reshape(-1,1)
                col_c = np.array(-ori_c[:,i]).reshape(-1,1)
                ori_A =  np.concatenate((ori_A, col_A),axis=1).reshape(constrain_num,-1)
                ori_c =  np.concatenate((ori_c, col_c),axis=1).reshape(1,-1)
                self.var_status[i] = i
                self.var_status = np.append(self.var_status, i)
        
        # add AUX variable based on constrain type 
        for i in range(len(cons_type_arr)):
            if (cons_type_arr[i] == -1):
                col_A = np.zeros(shape=(self.ori_constrain_num,1))
                col_A[i][0] = 1
                ori_A = np.concatenate((ori_A,col_A), axis=1).reshape(constrain_num,-1)
                col_c = np.zeros(shape=(1,1))    
                ori_c = np.concatenate((ori_c,col_c), axis=1).reshape(1,-1)
                self.var_status = np.append(self.var_status, self.NORMAL)
            if (cons_type_arr[i] == 1):
                ori_A[i,:] = -ori_A[i,:]
                b[i,:] = -b[i,:]
                col_A = np.zeros(shape=(self.ori_constrain_num,1))
                col_A[i][0] = 1
                ori_A = np.concatenate((ori_A,col_A), axis=1).reshape(constrain_num,-1)
                col_c = np.zeros(shape=(1,1))    
                ori_c = np.concatenate((ori_c,col_c), axis=1).reshape(1,-1)
                self.var_status = np.append(self.var_status, self.NORMAL)

        # if b[i] < 0, we need to choose the negative side of b in order to avoid dead-loop
        for i in range(ori_b.shape[0]):
            if (ori_b[i,:] < 0):
                ori_A[i,:] = -ori_A[i,:]
                ori_b[i,:] = -ori_b[i,:]
        
        self.c = ori_c
        self.A = ori_A
        self.b = ori_b
        self.res = ori_res
        self.res_type = 1
        self.constrain_num = (self.A).shape[0]
        self.var_num = (self.A).shape[1]
        
        if (self.debug == True):
            print("\nSimplex Table Init...")
            self.print_simplex_table([i for i in range(self.constrain_num)], [i+self.constrain_num for i in range(self.var_num-self.constrain_num)], self.A, np.empty(shape=(0,0)), self.c, np.empty(shape=(0,0)), self.b, self.res)
            print("Variable Status...")
            self.print_var_status(self.var_status)

    def print_var_status(self, var_status):
        if (self.debug == True):
            for i in range(len(self.var_status)):
                if (self.var_status[i] == self.NORMAL):
                    print("x{}".format(i),end="  ")
                elif (self.var_status[i] == self.NINF):
                    print("-x{}(x{})".format(i,i),end="  ")
                elif (self.var_status[i] == i):
                    print("x{}+(x{})".format(i,i),end="  ")
                elif (self.var_status[i] >= 0):
                    print("x{}-(x{})".format(self.var_status[i],i),end="  ")
            print("\n")

    def print_simplex_table(self, base_list, null_list, A_b, A_n, c_b, c_n, b, res):
        if (self.debug == True):
            # ===============
            print("+",end='')
            for i in range(len(base_list+null_list)+1):
                print("------",end='')
            print("+",end='')
            print("\n",end='')

            # base number
            print("|",end="")
            for i in range(len(base_list)):
                print("%5d"%base_list[i],end=" ")
            for i in range(len(null_list)):
                print("%5d"%null_list[i],end=" ")
            print("      |\n",end="")

            # ===============
            print("+",end='')
            for i in range(len(base_list+null_list)+1):
                print("------",end='')
            print("+",end='')
            print("\n",end='')
            
            # c arr line
            print("|",end="")
            for i in range(c_b.shape[1]):
                print("%5.2f"%c_b[0][i],end=" ")
            for i in range(c_n.shape[1]):
                print("%5.2f"%c_n[0][i],end=" ")
            print("|%5.2f"%res,end="")
            print("|\n",end="")

            # ===============
            print("+",end='')
            for i in range(len(base_list+null_list)+1):
                print("------",end='')
            print("+",end='')
            print("\n",end='')

            # A | b  matrix
            for i in range(self.constrain_num):
                print("|",end="")
                for j in range(A_b.shape[1]):
                    print("%5.2f"%A_b[i][j],end=" ")
                for j in range(A_n.shape[1]):
                    print("%5.2f"%A_n[i][j],end=" ")
                print("|%5.2f"%b[i],end="")
                print("|\n",end="")
            
            # ===============
            print("+",end='')
            for i in range(len(base_list+null_list)+1):
                print("------",end='')
            print("+",end='')
            print("\n",end='')        

    def split_matrix(self,mat,lista, listb):
        return mat[:,lista], mat[:,listb]

    def inv_matrix(self,mat):
        return np.linalg.inv(mat)
    
    def transform(self, A_b, A_n, c_b, c_n, b, res):
        res = res - float(np.dot(np.dot(c_b, self.inv_matrix(A_b)) , b))
        c_n = c_n - np.dot(np.dot(c_b, self.inv_matrix(A_b)), A_n)
        A_n = np.dot(self.inv_matrix(A_b), A_n)
        b   = np.dot(self.inv_matrix(A_b), b)
        A_b = np.eye(A_b.shape[0])
        c_b = np.zeros(shape=(c_b.shape))
        return A_b, A_n, c_b, c_n, b, res
    
    def find_inf_solution(self, A, c, row_num):
        # A_tmp = np.concatenate((A,c), axis=0)
        # incorrect!!!!!!!!!!!!!!!
        # return True if(np.linalg.matrix_rank(A_tmp) == row_num) else False
        return False

    def find_A_not_full_row_rank(self, A, row_num):
        return True if (np.linalg.matrix_rank(A) != row_num) else False

    def have_special_situation(self, A, c):
        find_inf_solution_res  =  self.find_inf_solution(A, c, A.shape[0])
        find_A_not_full_row_rank_res = self.find_A_not_full_row_rank(A, A.shape[0])
        if (find_inf_solution_res == True):
            return -1
        if (find_A_not_full_row_rank_res == True):
            return 0
        return 1

    def decide_end_simplex_or_not(self, c_n, b):
        return True if (np.max(c_n)<=1e-5 and np.min(b) >=-1e-5) else False

    def find_in_index(self, c_n, null_list):
        return null_list[np.argmax(c_n)]

    def find_out_index(self, A_n, b, in_index, null_list,base_list):
        in_index_pos_in_null_list = null_list.index(in_index)
        A_n_in_index_col = A_n[:,in_index_pos_in_null_list].reshape(-1,1)
        A_n_in_index_col_copy = A_n_in_index_col.copy()

        # since we might come across divide zero error, we change all 0 in the np array into -1
        # definitely we would not choose the 0 and the -1, so the effect is the same
        A_n_in_index_col_copy[A_n_in_index_col_copy == 0] = -1
        div_mat = b / A_n_in_index_col_copy

        # if the minimal elememnt in the div_mat has a negative corresponding A_n_in_index_col
        while(A_n_in_index_col_copy[np.argmin(div_mat)] <= 1e-5):
            if (np.min(div_mat) == self.INF):
                break
            div_mat[np.argmin(div_mat),:] = np.array(self.INF).reshape(1,-1)
        
        in_out_valid = True
        if (np.min(div_mat) == self.INF):
            in_out_valid = False

        return base_list[np.argmin(div_mat)], in_out_valid

    def change_base(self, base_list, null_list, A_b, A_n, c_b, c_n, b):
        in_index = self.find_in_index(c_n, null_list)
        out_index, in_out_valid = self.find_out_index(A_n, b, in_index, null_list, base_list)

        if (self.debug == True):
            print("IN={}, OUT={}".format(in_index, out_index))

        # find index in the list
        in_index_pos_in_null_list = null_list.index(in_index)
        out_index_pos_in_base_list = base_list.index(out_index)

        # swap two columns
        A_tmp = A_n[:, in_index_pos_in_null_list].copy()
        A_n[:, in_index_pos_in_null_list] = A_b[:, out_index_pos_in_base_list]
        A_b[:, out_index_pos_in_base_list] = A_tmp

        # swap tow columns
        c_tmp = c_n[:, in_index_pos_in_null_list].copy()
        c_n[:, in_index_pos_in_null_list] = c_b[:, out_index_pos_in_base_list]
        c_b[:, out_index_pos_in_base_list] = c_tmp

        # modify two lists
        null_list[in_index_pos_in_null_list] = out_index
        base_list[out_index_pos_in_base_list] = in_index

        return base_list, null_list, A_b, A_n, c_b, c_n, in_out_valid, in_index, out_index

    def simplex_loop(self,base_list, null_list, A_b, A_n, c_b, c_n, b, res):
        in_out_valid = True
        in_index = -1
        out_index = -1
        while (1):
            A_b, A_n, c_b, c_n, b, res = self.transform(A_b, A_n, c_b, c_n, b, res)
            if (self.debug == True):
                print("After transforming...")
            self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)
            if (self.debug == True):
                print("Judging finish simplex or not...  ",end="")
            end_of_simplex = self.decide_end_simplex_or_not(c_n, b)
            if (end_of_simplex == True): 
                if (self.debug == True):
                    print("Finally End!")
                break
            else:
                if (self.debug == True):
                    print("Keep Going!")
            if (self.debug == True):
                print("Changing base...  ",end="")
            
            prev_in_index = in_index
            prev_out_index = out_index
            base_list, null_list, A_b, A_n, c_b, c_n, in_out_valid, in_index, out_index = self.change_base(base_list, null_list, A_b, A_n, c_b, c_n, b)

            if (prev_in_index == out_index and prev_out_index == in_index):
                in_out_valid = False

            self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

            if (in_out_valid == False):
                break

        
        return base_list, null_list, A_b, A_n, c_b, c_n, b, res, in_out_valid


    def simplex_algo(self):
        # construct the initial simplex table
        # error check
        self.res_type = self.have_special_situation(self.A, self.c)

        if (self.res_type != 1):
            return self.res_type,0,[]
        
        # 1st stage for 2 stage simplex method
        base_list = [i+self.var_num for i in range(self.constrain_num)]
        null_list = [i for i in range(self.var_num)]
        A_n = self.A.copy()
        A_b = np.eye(self.constrain_num)
        c_n = np.zeros(shape=(1,self.var_num))
        c_b = np.full((1,self.constrain_num),-1)
        b   = self.b.copy()   
        res = 0

        if (self.debug == True):
            print("\n======1st Stage for Simplex Method======")
        self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)
        base_list, null_list, A_b, A_n, c_b, c_n, b, res, in_out_valid = self.simplex_loop(base_list, null_list, A_b, A_n, c_b, c_n, b, res)
        if (in_out_valid == False):
            self.res_type = 0
            return self.res_type, 0, []

        if (self.debug == True):
            print("\nInitial point FOUND!")
        self.print_simplex_table(base_list, null_list, A_b, A_n, c_b, c_n, b, res)

        # in the 1 stage of simplex method, we can find situation that can not find solution for simplex method
        if(abs(res-0) >= 1e-10):
            self.res_type = -1
            return self.res_type, 0, []
        
        self.null_list = [i for i in null_list if i < self.var_num]
        self.base_list = [i for i in base_list if i < self.var_num]
        if (len(self.base_list) < self.constrain_num):
            self.base_list += self.null_list[-(self.constrain_num - len(self.base_list)):]
            self.null_list  = self.null_list[:-(self.constrain_num - len(self.base_list))]
        self.A_b, self.A_n  = self.split_matrix(self.A, self.base_list, self.null_list)
        self.c_b, self.c_n  = self.split_matrix(self.c, self.base_list, self.null_list)
        
        if (self.debug == True):
            print("\n======2nd Stage for Simplex Method======")
        self.print_simplex_table(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)

        if (len(self.null_list) > 0):
            self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res, in_out_valid = self.simplex_loop(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)
            if (in_out_valid == False):
                self.res_type = 0
                return self.res_type, 0, []
        else:
            # I once come into a circumstances that null_list = [] and the inv of A_b is quite  strange and Ax = b cannnot be solved by A-1*b
            self.res = self.res - float(np.dot(np.dot(self.c_b, self.inv_matrix(self.A_b)) , self.b))
            self.b = np.dot(self.inv_matrix(self.A_b), self.b)

        if (self.debug == True):
            print("\nComplete, get final results!")
        self.print_simplex_table(self.base_list, self.null_list, self.A_b, self.A_n, self.c_b, self.c_n, self.b, self.res)
        
        self.res_type, self.opt_solution, self.solution = self.handle_result(self.res_type, self.base_list, self.null_list, (self.b).reshape(-1), self.res, self.var_status)

        return self.res_type, self.opt_solution, self.solution

    def std_simplex_algo(self):
        c = list(-self.std_c[0,:])
        A_ub = []
        A_eq = []
        b_ub = []
        b_eq = []
        for i in range(self.A.shape[0]): 
            if (self.cons_type_arr[i] == -1):
                A_ub.append(list(self.std_A[i]))
                b_ub.append(self.std_b[i][0])
            elif (self.cons_type_arr[i] == 0):
                A_eq.append(list(self.std_A[i]))
                b_eq.append(self.std_b[i][0])
            elif (self.cons_type_arr[i] == 1):
                A_ub.append(list(-self.std_A[i]))
                b_ub.append(-self.std_b[i][0])
        bound = []
        for i in range(len(self.var_type_arr)):
            if (self.var_type_arr[i] == 1):
                bound.append((0, None))
            if (self.var_type_arr[i] == -1):
                bound.append((None, 0))
            if (self.var_type_arr[i] == 0):
                bound.append((None, None))
        
        if (A_ub == []):
            res = linprog(c, A_ub=None, b_ub=None, A_eq=A_eq, b_eq=b_eq, bounds=bound, method='simplex')
        elif (A_eq == []):
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None, bounds=bound, method='simplex')
        else:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bound, method='simplex')

        res_type = 0
        if (res.status == 2 or res.status == 4):
            res_type = -1
        if (res.status == 3):
            res_type = 0
        if (res.status == 0 or res.status == 1):
            res_type = 1
        self.res_type = res_type
        self.opt_solution = res.fun
        self.solution = res.x
        return self.res_type, self.opt_solution, self.solution

    def handle_result(self, res_type, base_list, null_list, base_solution, opt_solution, var_status):
        # construct the solution list 
        base_cnt = 0
        null_cnt = 0
        solution = [0 for i in range(len(base_list+null_list))]

        for i in range(len(base_list)):
            solution[base_list[i]] = base_solution[i]

        # change solution list based on var_status
        for i in range(len(solution)):
            if (var_status[i] == self.NINF):
                solution[i] = -solution[i]
            elif (var_status[i] == self.NORMAL):
                continue
            elif (var_status[i] != i):
                solution[var_status[i]] -= solution[i]
        solution = solution[:self.ori_var_num]
        return res_type, opt_solution, solution
    
    def print_result(self):
        if (self.res_type == 1):
            print(self.res_type)
            print(self.opt_solution)
            for i in range(len(self.solution)-1):
                print(self.solution[i],end=" ")
            print(self.solution[len(self.solution)-1])
        elif (self.res_type == 0):
            print(self.res_type)
        elif (self.res_type == -1):
            print(self.res_type)
    


if __name__ == '__main__':
    test = True

    if (test == False):
        # 1st line of input
        # variable number and constrain number
        var_constrain_str = input()
        var_constrain_str_split = var_constrain_str.split()
        if (len(var_constrain_str_split) != 2 or int(var_constrain_str_split[0]) <=0 or int(var_constrain_str_split[1]) <= 0): 
            raise Exception('Variable and constrain number input ERROR!')
        var_num = int(var_constrain_str_split[0])
        constrain_num = int(var_constrain_str_split[1])

        # 2nd line of input
        # paramters for target function ( min -> by default)
        target_func_para_str = input()
        target_func_para_str_split = target_func_para_str.split()
        if (len(target_func_para_str_split) != var_num): 
            raise Exception('Parameters for target function input ERROR!')
        # change the (min) opt problem into (max) opt problem
        target_func_para_arr = [-int(i) for i in target_func_para_str_split]

        # init for important numpy variable
        # c must be a row vector
        c = np.array(target_func_para_arr).reshape(1,-1)
        # A,b must be empty numpy array
        A = np.empty(shape=(0, var_num))
        b = np.empty(shape=(0, 1))
        # res must be zero scalar
        res = 0

        # help the construction auxiliary variables
        constrain_equality = []
        # (constrain_number) lines lf input
        for i in range(constrain_num):
            constrain_str = input()
            constrain_str_split = constrain_str.split()
            if (len(constrain_str_split) != (var_num+2)): 
                raise Exception('input for constrain ERROR! constrain_str length = {}'.format(len(constrain_str.split())))

            constrain_para_arr = [int(i) for i in constrain_str_split]
            if (constrain_para_arr[-1] != -1 and constrain_para_arr[-1] != 0 and constrain_para_arr[-1] != 1): 
                raise Exception('input for constrain ERROR! last number in the constraint must be -1 or 1 or 0')

            # modify the constrain into standard forms ax+by <= c if it is ax+by >= c
            if (constrain_para_arr[-1] == 1):
                constrain_para_arr = [-i for i in constrain_para_arr]

            constrain_equality.append(constrain_para_arr[-1])

            row_A = np.array(constrain_para_arr[:-2]).reshape(-1,var_num)
            row_b = np.array(constrain_para_arr[-2]).reshape(-1,1)
            A = np.concatenate((A, row_A),axis=0).reshape(-1, var_num)
            b = np.concatenate((b, row_b),axis=0).reshape(-1, 1)

        # one lines for variable limitation
        var_constrain_str = input()
        var_constrain_str_split = var_constrain_str.split()
        if (len(var_constrain_str_split) != var_num):
            raise Exception('input for constraints on variables ERROR!')

        var_constrain_arr = [int(i) for i in var_constrain_str_split]
        
        simplex = Simplex_Table(c,A,b,res,var_constrain_arr,constrain_equality,debug=False)
        res_type, opt_solution, solution = simplex.simplex_algo()
        simplex.print_result()
        res_type, opt_solution, solution = simplex.std_simplex_algo()
        simplex.print_result()

    elif (test == True):
        iter_num = 0
        while(1):
            iter_num += 1
            np.random.seed(random.randint(0,23453)+iter_num % 34224529)
            constrain_num = 8
            variable_num = 10
            c = np.random.randn(1,variable_num)
            A = np.random.randn(constrain_num, variable_num)
            while (np.linalg.matrix_rank(A) != constrain_num):
                A = np.random.randn(constrain_num, variable_num)
            b = np.random.randn(constrain_num, 1)

            res = 0
            var_constrain_arr = []
            constrain_equality = []
            for i in range(variable_num):
                var_constrain_arr.append(np.random.randint(-1,2))
            for i in range(constrain_num):
                constrain_equality.append(np.random.randint(-1,2))

            simplex = Simplex_Table(c,A,b,res,var_constrain_arr,constrain_equality,debug=False)
            std_res_type, std_opt_solution, std_solution = simplex.std_simplex_algo()
            #if (std_res_type != 1):
            #    continue
            simplex.print_result()
            res_type, opt_solution, solution = simplex.simplex_algo()
            simplex.print_result()

            if (res_type != std_res_type):
                raise Exception('answer type error')
            if (res_type == 1 and abs(opt_solution - std_opt_solution)>1e-5):
                raise Exception('optimal solution error')
            if (res_type == 1):
                var_check = True
                for i in range(len(solution)):
                    if (abs(solution[i] - std_solution[i]) > 1e-5):
                        var_check = False
                        break
                if (var_check == False):
                    print('variable solution error')
        

