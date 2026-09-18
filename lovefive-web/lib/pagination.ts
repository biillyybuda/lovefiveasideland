type Page<T> = { data: T[] | null; error: { message: string } | null };

/** Continue until an empty page, even when the server's cap is below our request. */
export async function loadAllPages<T>(fetchPage: (from: number, to: number) => PromiseLike<Page<T>>): Promise<T[]> {
  const rows: T[] = [];
  for (;;) {
    const { data, error } = await fetchPage(rows.length, rows.length + 499);
    if (error) throw new Error(error.message);
    if (!data?.length) return rows;
    rows.push(...data);
  }
}
