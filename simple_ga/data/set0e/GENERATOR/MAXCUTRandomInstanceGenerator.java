import java.io.*;
import java.util.Random;

public class MAXCUTRandomInstanceGenerator
{
  public static double computeEdgeWeight( int i, int j, double vertex_coordinates[][] )
  {
    double result, delta_x, delta_y;

    delta_x = vertex_coordinates[j][0]-vertex_coordinates[i][0];
    delta_y = vertex_coordinates[j][1]-vertex_coordinates[i][1];
    result  = Math.sqrt( delta_x*delta_x + delta_y*delta_y );

    return( result );
  }

  public static int[] mergeSort( double array[] )
  {
    int i, sorted[], tosort[];
  
    sorted = new int[array.length];
    tosort = new int[array.length];
    for( i = 0; i < array.length; i++ )
      tosort[i] = i;
  
    if( array.length == 1 )
      sorted[0] = 0;
    else
      mergeSortWithinBounds( array, sorted, tosort, 0, array.length-1 );
  
    return( sorted );
  }
  
  public static void mergeSortWithinBounds( double array[], int sorted[], int tosort[], int p, int q )
  {
    int r;
  
    if( p < q )
    {
      r = (p + q) / 2;
      mergeSortWithinBounds( array, sorted, tosort, p, r );
      mergeSortWithinBounds( array, sorted, tosort, r+1, q );
      mergeSortMerge( array, sorted, tosort, p, r+1, q );
    }
  }
  
  public static void mergeSortMerge( double array[], int sorted[], int tosort[], int p, int r, int q )
  {
    boolean first;
    int     i, j, k;
  
    i = p;
    j = r;
    for( k = p; k <= q; k++ )
    {
      first = false;
      if( j <= q )
      {
        if( i < r )
        {
          if( array[tosort[i]] < array[tosort[j]] )
            first = true;
        }
      }
      else
        first = true;
  
      if( first )
      {
        sorted[k] = tosort[i];
        i++;
      }
      else
      {
        sorted[k] = tosort[j];
        j++;
      }
    }
  
    for( k = p; k <= q; k++ )
      tosort[k] = sorted[k];
  }

  public static void main( String ps[] )
  {
    boolean edge_selection_matrix[][];
    int     i, j, number_of_vertices, sqrt_number_of_vertices, number_of_edges, weight, sorted[];
    double  vertex_coordinates[][], edge_weight_matrix[][];
    Random random;

    random             = new Random();

    number_of_vertices = Integer.valueOf( ps[0] ).intValue();

    vertex_coordinates = new double[number_of_vertices][2];

    for( i = 0; i < number_of_vertices; i++ )
    {
      vertex_coordinates[i][0] = random.nextDouble()*1000;
      vertex_coordinates[i][1] = random.nextDouble()*1000;
    }

    edge_weight_matrix = new double[number_of_vertices][number_of_vertices];
    for( i = 0; i < number_of_vertices; i++ )
    {
      edge_weight_matrix[0][0] = 0;
      for( j = i+1; j < number_of_vertices; j++ )
      {
        edge_weight_matrix[i][j] = computeEdgeWeight( i, j, vertex_coordinates );
        edge_weight_matrix[j][i] = edge_weight_matrix[i][j];
      }
    }

    edge_selection_matrix = new boolean[number_of_vertices][number_of_vertices];
    for( i = 0; i < number_of_vertices; i++ )
      for( j = 0; j < number_of_vertices; j++ )
        edge_selection_matrix[i][j] = false;

    sqrt_number_of_vertices = (int) Math.sqrt( number_of_vertices );
    for( i = 0; i < number_of_vertices; i++ )
    {
      sorted = mergeSort( edge_weight_matrix[i] );
      for( j = 0; j < sqrt_number_of_vertices; j++ )
      {
        edge_selection_matrix[i][sorted[j+1]] = true;
        edge_selection_matrix[sorted[j+1]][i] = true;
      }
    }

    number_of_edges = 0;
    for( i = 0; i < number_of_vertices; i++ )
    {
      for( j = i+1; j < number_of_vertices; j++ )
      {
        if( edge_selection_matrix[i][j] )
	  number_of_edges++;
      }
    }

    System.out.println(number_of_vertices + " " + number_of_edges);
    for( i = 0; i < number_of_vertices; i++ )
    {
      for( j = i+1; j < number_of_vertices; j++ )
      {
        if( edge_selection_matrix[i][j] )
          System.out.println((i+1) + " " + (j+1) + " " + ((int) edge_weight_matrix[i][j]));
      }
    }
  }
}
