import java.io.*;
import java.util.Random;

public class MAXCUTRandomInstanceGenerator
{
  public static int computeEdgeWeight( int i, int j, double vertex_coordinates[][] )
  {
    int    result;
    double delta_x, delta_y;

    delta_x = vertex_coordinates[j][0]-vertex_coordinates[i][0];
    delta_y = vertex_coordinates[j][1]-vertex_coordinates[i][1];
    result  = (int) Math.sqrt( delta_x*delta_x + delta_y*delta_y );

    return( result );
  }

  public static void main( String ps[] )
  {
    int    i, j, number_of_vertices, number_of_edges, weight;
    double vertex_coordinates[][];
    Random random;

    random             = new Random();

    number_of_vertices = Integer.valueOf( ps[0] ).intValue();
    number_of_edges    = ((number_of_vertices*(number_of_vertices-1))/2);

    vertex_coordinates = new double[number_of_vertices][2];

    for( i = 0; i < number_of_vertices; i++ )
    {
      vertex_coordinates[i][0] = random.nextDouble()*1000;
      vertex_coordinates[i][1] = random.nextDouble()*1000;
    }

    System.out.println(number_of_vertices + " " + number_of_edges);
    for( i = 0; i < number_of_vertices; i++ )
    {
      for( j = i+1; j < number_of_vertices; j++ )
      {
        weight = computeEdgeWeight( i, j, vertex_coordinates );
        System.out.println((i+1) + " " + (j+1) + " " + weight);
      }
    }
  }
}
