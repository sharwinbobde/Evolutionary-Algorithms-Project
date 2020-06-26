import java.io.*;
import java.util.Random;

public class MAXCUTRandomInstanceGenerator
{
  public static int generateEdgeWeight( Random random )
  {
    int    result;
    double z1, z2;

    z1     = -Math.log(random.nextDouble());
    z2     = -Math.log(random.nextDouble());
    result = (int) ((z1/(z1+z2))*5)+1;
    result = result >= 6 ? 5 : result;

    return( result );
  }

  public static void main( String ps[] )
  {
    int    i, j, number_of_vertices, number_of_edges, weight;
    Random random;

    random             = new Random();

    number_of_vertices = Integer.valueOf( ps[0] ).intValue();
    number_of_edges    = ((number_of_vertices*(number_of_vertices-1))/2);

    System.out.println(number_of_vertices + " " + number_of_edges);
    for( i = 0; i < number_of_vertices; i++ )
    {
      for( j = i+1; j < number_of_vertices; j++ )
      {
        weight = generateEdgeWeight( random );
        System.out.println((i+1) + " " + (j+1) + " " + weight);
      }
    }
  }
}
